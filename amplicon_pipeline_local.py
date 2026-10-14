#!/usr/bin/env python3
"""
amplicon_pipeline_local.py
--------------------------
Local (non-PBS) version of submit_amplicon_pipeline.sh + amplicon_pipeline.pbs.

For every locus in primers.txt, runs   cutadapt -> bbduk -> bbmerge
over every sample in sequences.txt, producing exactly the same directory
layout, per-locus summary TSVs, command logs and per-tool logs as the PBS
version, so plot_pipeline_summary.py works unchanged on the output.

Requirements: cutadapt on PATH, java on PATH (for BBTools), BBTools.

Usage (from WSL, in the project folder):
    python3 amplicon_pipeline_local.py
    python3 amplicon_pipeline_local.py -b /mnt/c/FB -i /mnt/c/FB/RAW \
        -s /mnt/c/FB/RAW/sequences.txt -p /mnt/c/FB/primers.txt \
        --bbmap /mnt/c/LP/bbmap/bbmap.sh -t 8 -j 4 --plot

Run with -h for all options.
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import threading

# ---------------------------------------------------------------------------
# Defaults (mirror the PBS scripts, with local paths)
# ---------------------------------------------------------------------------
DEFAULT_BBMAP = "/mnt/c/LP/bbmap/bbmap.sh"
DEFAULT_ERROR_RATE = 0.1
DEFAULT_OVERLAP = 3
RELAXED_ERROR_RATE = 0.2
RELAXED_OVERLAP = 10

# R1 suffixes recognised when detecting read format (same order as the .sh)
R1_PATTERNS = [
    "_R1_001.fastq.gz", "_R1.fastq.gz", "_1.fastq.gz",
    "_R1_001.fq.gz",    "_R1.fq.gz",    "_1.fq.gz",
]
# Suffixes stripped from sequences.txt entries to get sample names (as .pbs)
STRIP_SUFFIXES = [
    "_R1_001.fastq.gz", "_R2_001.fastq.gz",
    "_R1.fastq.gz",     "_R2.fastq.gz",
    "_1.fastq.gz",      "_2.fastq.gz",
    "_R1_001.fq.gz",    "_R2_001.fq.gz",
    "_R1.fq.gz",        "_R2.fq.gz",
    "_1.fq.gz",         "_2.fq.gz",
    ".R1.clean.fq.gz",  ".R2.clean.fq.gz",
]

_print_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Logging: everything printed also goes to <base>/logs/amplicon_local_*.log
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def say(*args):
    with _print_lock:
        print(*args, flush=True)


def die(msg):
    sys.exit(f"ERROR: {msg}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_RC = str.maketrans("ACGTRYSWKMBDHVNIacgtryswkmbdhvni",
                    "TGCAYRSWMKVHDBNItgcayrswmkvhdbni")


def rc(seq: str) -> str:
    """Reverse complement, full IUPAC including inosine I."""
    return seq.translate(_RC)[::-1]


def is_degenerate(seq: str) -> bool:
    return re.search(r"[RYSWKMBDHVNI]", seq, re.I) is not None


def has_reads(path: str) -> bool:
    """True if a gzipped fastq exists and contains at least one record."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        with gzip.open(path, "rt") as fh:
            return bool(fh.readline().strip())
    except (OSError, EOFError):
        return False


def write_empty_gz(path: str) -> None:
    with gzip.open(path, "wb"):
        pass


def read_lines(path: str) -> list[str]:
    """Read non-empty lines, tolerating CRLF / BOM / trailing whitespace."""
    with open(path, encoding="utf-8-sig") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def load_primers(path: str) -> list[dict]:
    primers = []
    for lineno, line in enumerate(read_lines(path), 1):
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            die(f"{path} line {lineno}: expected 5 fields "
                f"(locus fwd_name fwd_seq rev_name rev_seq), got {len(parts)}:\n  {line}")
        locus, fwd_name, fwd_seq, rev_name, rev_seq = parts
        primers.append(dict(locus=locus, fwd_name=fwd_name, fwd_seq=fwd_seq,
                            rev_name=rev_name, rev_seq=rev_seq,
                            wildcard=is_degenerate(fwd_seq) or is_degenerate(rev_seq)))
    if not primers:
        die(f"no primers found in {path}")
    return primers


def detect_read_format(entries: list[str]) -> tuple[str, str]:
    first = entries[0]
    for pat in R1_PATTERNS:
        if first.endswith(pat):
            r1 = pat
            r2 = r1.replace("_R1", "_R2") if "_R1" in r1 else r1.replace("_1.", "_2.")
            return r1, r2
    # First entry may be an R2 name — try that too
    for pat in R1_PATTERNS:
        r2pat = pat.replace("_R1", "_R2") if "_R1" in pat else pat.replace("_1.", "_2.")
        if first.endswith(r2pat):
            return pat, r2pat
    die(f"could not detect read format from first entry in sequences file: {first}\n"
        f"  Expected a suffix like _R1.fq.gz, _R1_001.fastq.gz, _1.fq.gz, etc.")


def sample_names(entries: list[str]) -> list[str]:
    seen, out = set(), []
    for name in entries:
        name = os.path.basename(name)
        for suf in STRIP_SUFFIXES:
            if name.endswith(suf):
                name = name[: -len(suf)]
                break
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def run(cmd: list[str], logfile: str, cmds_file: str, append_log=False) -> int:
    """Run a command, write combined stdout/stderr to logfile, record the cmd."""
    with _print_lock:
        with open(cmds_file, "a") as fh:
            fh.write(" ".join(cmd) + "\n")
    with open(logfile, "a" if append_log else "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def grep(pattern: str, path: str):
    if not os.path.isfile(path):
        return None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = re.search(pattern, line)
            if m:
                return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Per-sample stage functions — each returns a TSV row (list of str)
# ---------------------------------------------------------------------------
def cutadapt_sample(sample, cfg, L, dirs):
    in_r1 = os.path.join(cfg.in_path, sample + cfg.fmt_r1)
    in_r2 = os.path.join(cfg.in_path, sample + cfg.fmt_r2)
    out_r1 = os.path.join(dirs["ca"], f"{sample}_R1.fq.gz")
    out_r2 = os.path.join(dirs["ca"], f"{sample}_R2.fq.gz")
    js = os.path.join(dirs["ca"], f"{sample}_cutadapt.json")
    log = os.path.join(dirs["log"], f"cutadapt_{L['locus']}_{sample}.log")

    if not (os.path.isfile(in_r1) and os.path.isfile(in_r2)):
        say(f"  WARNING [{L['locus']}] raw files not found, skipping {sample}\n"
            f"    R1: {in_r1}\n    R2: {in_r2}")
        write_empty_gz(out_r1)
        write_empty_gz(out_r2)
        return [sample, "NA", "NA", "NA"], False

    done = cfg.resume and os.path.isfile(js) and os.path.isfile(out_r1)
    if not done:
        cmd = ["cutadapt",
               "-e", str(cfg.error_rate),
               "--overlap", str(cfg.overlap),
               "--minimum-length", "50",
               "--discard-untrimmed",
               "-n", "2",
               "-g", "^" + L["fwd_seq"],
               "-a", rc(L["rev_seq"]),
               "-G", "^" + L["rev_seq"],
               "-A", rc(L["fwd_seq"]),
               "-o", out_r1, "-p", out_r2,
               f"--json={js}",
               "-j", str(cfg.threads_per_job)]
        if L["wildcard"]:
            cmd.append("--match-read-wildcards")
        cmd += [in_r1, in_r2]
        if os.path.exists(js):
            os.remove(js)
        rcode = run(cmd, log, dirs["ca_cmds"])
        if rcode != 0 or not os.path.isfile(js):
            say(f"  ERROR [{L['locus']}] cutadapt failed for {sample} (exit {rcode}) — see {log}")
            write_empty_gz(out_r1)
            write_empty_gz(out_r2)
            return [sample, "NA", "NA", "NA"], True

    with open(js) as fh:
        rcounts = json.load(fh)["read_counts"]
    inp, out = rcounts["input"], rcounts["output"]
    pct = f"{out / inp * 100:.1f}" if inp else "0.0"
    say(f"  cutadapt [{L['locus']}] {sample}: {inp:,} in -> {out:,} written ({pct}%)"
        + ("  (resumed)" if done else ""))
    return [sample, str(inp), str(out), pct], False


def bbduk_sample(sample, cfg, L, dirs):
    in_r1 = os.path.join(dirs["ca"], f"{sample}_R1.fq.gz")
    in_r2 = os.path.join(dirs["ca"], f"{sample}_R2.fq.gz")
    out_r1 = os.path.join(dirs["bbd"], f"{sample}_R1.fq.gz")
    out_r2 = os.path.join(dirs["bbd"], f"{sample}_R2.fq.gz")
    log = os.path.join(dirs["log"], f"bbduk_{L['locus']}_{sample}.log")

    if not has_reads(in_r1):
        say(f"  bbduk    [{L['locus']}] {sample}: 0 pairs from cutadapt — skipped")
        write_empty_gz(out_r1)
        write_empty_gz(out_r2)
        return [sample, "0", "0", "0.0"], False

    done = (cfg.resume and os.path.isfile(out_r1)
            and grep(r"^Result:\s+([0-9]+)\s+reads", log) is not None)
    failed = False
    if not done:
        cmd = ["bash", cfg.bbduk,
               f"in1={in_r1}", f"in2={in_r2}",
               f"out1={out_r1}", f"out2={out_r2}",
               "minlen=80", "ktrim=r", "k=23", "mink=8", "hdist=1", "tbo", "tpe",
               f"ref={cfg.adapters}",
               "maxns=0", "qtrim=r", "trimq=10", "interleaved=f",
               f"threads={cfg.threads_per_job}", "overwrite=t"]
        if cfg.mem:
            cmd.append(f"-Xmx{cfg.mem}")
        rcode = run(cmd, log, dirs["bbd_cmds"])
        if rcode != 0:
            say(f"  ERROR [{L['locus']}] bbduk failed for {sample} (exit {rcode}) — see {log}")
            write_empty_gz(out_r1)
            write_empty_gz(out_r2)
            failed = True

    inp = grep(r"^Input:\s+([0-9]+)\s+reads", log) or "NA"
    out = grep(r"^Result:\s+([0-9]+)\s+reads", log) or "NA"
    pct = grep(r"^Result:.*\(([\d.]+)%", log) or "NA"
    say(f"  bbduk    [{L['locus']}] {sample}: {inp} in -> {out} remaining ({pct}%)"
        + ("  (resumed)" if done else ""))
    return [sample, inp, out, pct], failed


def bbmerge_sample(sample, cfg, L, dirs):
    in_r1 = os.path.join(dirs["bbd"], f"{sample}_R1.fq.gz")
    in_r2 = os.path.join(dirs["bbd"], f"{sample}_R2.fq.gz")
    merged = os.path.join(dirs["merged"], f"{sample}_merged.fq.gz")
    unm_r1 = os.path.join(dirs["merged"], f"{sample}_unmerged_R1.fq.gz")
    unm_r2 = os.path.join(dirs["merged"], f"{sample}_unmerged_R2.fq.gz")
    hist = os.path.join(dirs["hist"], f"{sample}.hist")
    log = os.path.join(dirs["log"], f"bbmerge_{L['locus']}_{sample}.log")

    if not has_reads(in_r1):
        say(f"  bbmerge  [{L['locus']}] {sample}: 0 pairs from bbduk — skipped")
        return [sample, "0", "0", "0.0"], False

    done = (cfg.resume and os.path.isfile(merged)
            and grep(r"^Joined:\s+([0-9]+)", log) is not None)
    failed = False
    if not done:
        cmd = ["bash", cfg.bbmerge,
               f"in1={in_r1}", f"in2={in_r2}",
               f"out={merged}",
               f"outu1={unm_r1}", f"outu2={unm_r2}",
               f"hist={hist}",
               f"threads={cfg.threads_per_job}", "overwrite=t"]
        if cfg.mem:
            cmd.append(f"-Xmx{cfg.mem}")
        rcode = run(cmd, log, dirs["bbm_cmds"])
        if rcode != 0:
            say(f"  ERROR [{L['locus']}] bbmerge failed for {sample} (exit {rcode}) — see {log}")
            failed = True

    pairs = grep(r"^Pairs:\s+([0-9]+)", log) or "NA"
    joined = grep(r"^Joined:\s+([0-9]+)", log) or "NA"
    pct = grep(r"^Joined:\s+[0-9]+\s+([\d.]+)%", log) or "0.0"
    say(f"  bbmerge  [{L['locus']}] {sample}: {pairs} pairs -> {joined} merged ({pct}%)"
        + ("  (resumed)" if done else ""))
    return [sample, pairs, joined, pct], failed


# ---------------------------------------------------------------------------
# Run one stage across all samples (optionally N samples in parallel)
# ---------------------------------------------------------------------------
def run_stage(func, samples, cfg, L, dirs, tsv_path, header):
    failures = []
    if cfg.jobs > 1:
        with cf.ThreadPoolExecutor(max_workers=cfg.jobs) as ex:
            results = list(ex.map(lambda s: func(s, cfg, L, dirs), samples))
    else:
        results = [func(s, cfg, L, dirs) for s in samples]
    with open(tsv_path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for row, failed in results:
            fh.write("\t".join(row) + "\n")
            if failed:
                failures.append(row[0])
    return failures


def run_locus(L, samples, cfg):
    locus = L["locus"]
    base = cfg.base_path
    dirs = {
        "ca": os.path.join(base, f"{locus}_cutadapt"),
        "bbd": os.path.join(base, f"{locus}_bbduk"),
        "merged": os.path.join(base, f"{locus}_merged"),
        "hist": os.path.join(base, f"{locus}_merge_hist"),
        "cmd": os.path.join(base, "commands"),
        "log": os.path.join(base, "logs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    dirs["ca_cmds"] = os.path.join(dirs["cmd"], f"cutadapt_{locus}.cmds")
    dirs["bbd_cmds"] = os.path.join(dirs["cmd"], f"bbduk_{locus}.cmds")
    dirs["bbm_cmds"] = os.path.join(dirs["cmd"], f"bbmerge_{locus}.cmds")
    for f in (dirs["ca_cmds"], dirs["bbd_cmds"], dirs["bbm_cmds"]):
        open(f, "w").close()

    bar = "=" * 70
    say(f"\n{bar}\n[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] LOCUS {locus}\n"
        f"  fwd: {L['fwd_name']} = {L['fwd_seq']}\n"
        f"  rev: {L['rev_name']} = {L['rev_seq']}\n"
        f"  wildcard: {int(L['wildcard'])}\n{bar}")

    fails = {}
    say(f"\n--- STAGE 1: cutadapt ({locus}) ---")
    fails["cutadapt"] = run_stage(
        cutadapt_sample, samples, cfg, L, dirs,
        os.path.join(base, f"cutadapt_{locus}_summary.tsv"),
        ["Sample", "Input_Pairs", "Pairs_Written", "Pct_Written"])

    say(f"\n--- STAGE 2: bbduk ({locus}) ---")
    fails["bbduk"] = run_stage(
        bbduk_sample, samples, cfg, L, dirs,
        os.path.join(base, f"bbduk_{locus}_summary.tsv"),
        ["Sample", "Input_Reads", "Result_Reads", "Pct_Remaining"])

    say(f"\n--- STAGE 3: bbmerge ({locus}) ---")
    fails["bbmerge"] = run_stage(
        bbmerge_sample, samples, cfg, L, dirs,
        os.path.join(base, f"bbmerge_{locus}_summary.tsv"),
        ["Sample", "Pairs", "Joined", "Pct_Joined"])

    say(f"\n[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Done — {locus}")
    return {k: v for k, v in fails.items() if v}


# ---------------------------------------------------------------------------
# Argument parsing / setup
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Run the amplicon pipeline (cutadapt -> bbduk -> bbmerge) locally, "
                    "one locus after another.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="primers.txt format (whitespace-delimited, one locus per line):\n"
               "  locus_name  fwd_name  fwd_seq  rev_name  rev_seq")
    p.add_argument("-b", "--base-path", default=os.getcwd(), metavar="DIR",
                   help="Root output directory [default: current directory]")
    p.add_argument("-i", "--in-path", default=None, metavar="DIR",
                   help="Raw reads directory [default: <base_path>/RAW, or <base_path>/raw]")
    p.add_argument("-s", "--sequences", default=None, metavar="FILE",
                   help="sequences.txt [default: ./sequences.txt, else <in_path>/sequences.txt, "
                        "else built from the R1 files in <in_path>]")
    p.add_argument("-p", "--primers", default="primers.txt", metavar="FILE",
                   help="primers.txt [default: ./primers.txt]")
    p.add_argument("--bbmap", default=DEFAULT_BBMAP, metavar="PATH",
                   help=f"bbmap.sh (or the bbmap directory); bbduk.sh and bbmerge.sh "
                        f"are taken from the same folder [default: {DEFAULT_BBMAP}]")
    p.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 4,
                   help="Total CPU threads to use [default: all cores, %(default)s]")
    p.add_argument("-j", "--jobs", type=int, default=1,
                   help="Samples processed in parallel; threads are split between them. "
                        "Small amplicon files are dominated by tool start-up time, so "
                        "e.g. -j 4 is usually much faster than -j 1 [default: 1]")
    p.add_argument("--mem", default=None, metavar="SIZE",
                   help="Java heap for bbduk/bbmerge, e.g. 2g [default: BBTools auto; "
                        "2g when --jobs > 1]")
    p.add_argument("-e", "--error-rate", type=float, default=None,
                   help=f"cutadapt -e primer error rate [default: {DEFAULT_ERROR_RATE}]")
    p.add_argument("-O", "--overlap", type=int, default=None,
                   help=f"cutadapt --overlap minimum primer overlap (bp) [default: {DEFAULT_OVERLAP}]")
    p.add_argument("-r", "--relaxed", action="store_true",
                   help=f"Relaxed preset: -e {RELAXED_ERROR_RATE} -O {RELAXED_OVERLAP}, "
                        "outputs to <base_path>/relaxed/ (explicit -e/-O still win)")
    p.add_argument("-l", "--loci", nargs="+", default=None, metavar="LOCUS",
                   help="Only run these loci from primers.txt [default: all]")
    p.add_argument("--resume", action="store_true",
                   help="Skip sample/stage steps whose outputs and logs already exist")
    p.add_argument("--plot", action="store_true",
                   help="Run plot_pipeline_summary.py on the output when finished")
    p.add_argument("-m", "--metadata", default=None, metavar="FILE",
                   help="Metadata CSV passed to plot_pipeline_summary.py (with --plot)")
    p.add_argument("--dry-run", action="store_true",
                   help="Check inputs and print the settings, then stop")
    return p.parse_args()


def setup(args):
    # cutadapt settings & relaxed preset
    args.error_rate = args.error_rate if args.error_rate is not None else (
        RELAXED_ERROR_RATE if args.relaxed else DEFAULT_ERROR_RATE)
    args.overlap = args.overlap if args.overlap is not None else (
        RELAXED_OVERLAP if args.relaxed else DEFAULT_OVERLAP)
    if args.error_rate > 0.3:
        print(f"WARNING: error rate {args.error_rate} is very high (>0.3). It is a proportion "
              f"of primer length (20 bp primer: 0.1 -> 2 mismatches, 0.2 -> 4).")

    orig_base = os.path.abspath(args.base_path)
    args.base_path = os.path.join(orig_base, "relaxed") if args.relaxed else orig_base

    # Raw reads (always relative to the original base, as in the .sh)
    if args.in_path is None:
        for cand in ("RAW", "raw"):
            if os.path.isdir(os.path.join(orig_base, cand)):
                args.in_path = os.path.join(orig_base, cand)
                break
        else:
            die(f"no raw reads directory found at {orig_base}/RAW or {orig_base}/raw (use -i)")
    args.in_path = os.path.abspath(args.in_path)
    if not os.path.isdir(args.in_path):
        die(f"input directory not found: {args.in_path}")

    # Primers
    args.primers = os.path.abspath(args.primers)
    if not os.path.isfile(args.primers):
        die(f"primers file not found: {args.primers}")

    # Sequences file
    if args.sequences is None:
        for cand in (os.path.abspath("sequences.txt"),
                     os.path.join(args.in_path, "sequences.txt")):
            if os.path.isfile(cand):
                args.sequences = cand
                break
    if args.sequences is not None:
        args.sequences = os.path.abspath(args.sequences)
        if not os.path.isfile(args.sequences):
            die(f"sequences file not found: {args.sequences}")
        entries = read_lines(args.sequences)
    else:
        entries = sorted(f for f in os.listdir(args.in_path)
                         if any(f.endswith(p) for p in R1_PATTERNS))
        if not entries:
            die(f"no sequences.txt given/found and no R1 fastq files in {args.in_path}")
        args.sequences = f"(built from {len(entries)} R1 files in {args.in_path})"
    if not entries:
        die(f"sequences file is empty: {args.sequences}")
    args.fmt_r1, args.fmt_r2 = detect_read_format(entries)
    args.samples = sample_names(entries)

    # BBTools
    bb = os.path.abspath(args.bbmap)
    bbdir = bb if os.path.isdir(bb) else os.path.dirname(bb)
    args.bbduk = os.path.join(bbdir, "bbduk.sh")
    args.bbmerge = os.path.join(bbdir, "bbmerge.sh")
    args.adapters = os.path.join(bbdir, "resources", "adapters.fa")
    for f in (args.bbduk, args.bbmerge, args.adapters):
        if not os.path.isfile(f):
            die(f"BBTools file not found: {f}  (check --bbmap)")

    # Tools on PATH
    if shutil.which("cutadapt") is None:
        die("cutadapt not found on PATH")
    if shutil.which("java") is None:
        die("java not found on PATH (BBTools needs it)")
    args.cutadapt_version = subprocess.run(
        ["cutadapt", "--version"], capture_output=True, text=True).stdout.strip()

    # Threads / jobs
    args.jobs = max(1, args.jobs)
    args.threads_per_job = max(1, args.threads // args.jobs)
    if args.jobs > 1 and not args.mem:
        args.mem = "2g"

    # Loci
    primers = load_primers(args.primers)
    if args.loci:
        known = {p["locus"] for p in primers}
        missing = [l for l in args.loci if l not in known]
        if missing:
            die(f"loci not in primers.txt: {', '.join(missing)} "
                f"(available: {', '.join(sorted(known))})")
        primers = [p for p in primers if p["locus"] in args.loci]
    args.primer_list = primers
    args.orig_base = orig_base
    return args


def run_plot(cfg):
    here = os.path.dirname(os.path.abspath(__file__))
    script = None
    for cand in (os.path.join(here, "plot_pipeline_summary.py"),
                 os.path.join(cfg.orig_base, "plot_pipeline_summary.py")):
        if os.path.isfile(cand):
            script = cand
            break
    if script is None:
        say("WARNING: --plot given but plot_pipeline_summary.py not found next to this "
            "script or in the base path — skipping plots.")
        return
    # Remove stale combined TSVs so the plotter rebuilds them from this run
    for stage in ("cutadapt", "bbduk", "bbmerge"):
        f = os.path.join(cfg.base_path, f"{stage}_summary.tsv")
        if os.path.isfile(f):
            os.remove(f)
    cmd = [sys.executable, script, "-b", cfg.base_path, "-p", cfg.primers]
    if cfg.metadata:
        cmd += ["-m", os.path.abspath(cfg.metadata)]
    say(f"\nRunning plots: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        say(line.rstrip("\n"))
    if proc.wait() != 0:
        say(f"WARNING: plot_pipeline_summary.py exited with code {proc.returncode}")


def main():
    cfg = setup(parse_args())

    os.makedirs(os.path.join(cfg.base_path, "logs"), exist_ok=True)
    runlog = os.path.join(cfg.base_path, "logs",
                          f"amplicon_local_{dt.datetime.now():%Y%m%d_%H%M%S}.log")
    logfh = open(runlog, "w")
    sys.stdout = Tee(sys.__stdout__, logfh)

    say("Amplicon pipeline (local)")
    say(f"  base_path   : {cfg.base_path}")
    say(f"  in_path     : {cfg.in_path}")
    say(f"  seq_file    : {cfg.sequences}")
    say(f"  primers     : {cfg.primers}")
    say(f"  samples     : {len(cfg.samples)}")
    say(f"  loci        : {', '.join(p['locus'] for p in cfg.primer_list)}")
    say(f"  read format : R1={cfg.fmt_r1}  R2={cfg.fmt_r2}")
    say(f"  cutadapt    : {cfg.cutadapt_version}  -e {cfg.error_rate}  -O {cfg.overlap} bp")
    say(f"  bbtools     : {os.path.dirname(cfg.bbduk)}")
    say(f"  threads     : {cfg.threads} total, {cfg.jobs} sample(s) at a time, "
        f"{cfg.threads_per_job} per tool" + (f", java -Xmx{cfg.mem}" if cfg.mem else ""))
    if cfg.relaxed:
        say(f"  *** RELAXED PRESET ACTIVE — outputs in {cfg.base_path} ***")
    if cfg.resume:
        say("  resume      : on (completed steps are skipped)")
    say(f"  run log     : {runlog}")

    missing = [s for s in cfg.samples
               if not os.path.isfile(os.path.join(cfg.in_path, s + cfg.fmt_r1))
               or not os.path.isfile(os.path.join(cfg.in_path, s + cfg.fmt_r2))]
    if missing:
        say(f"  WARNING: {len(missing)} sample(s) missing R1/R2 in {cfg.in_path}, "
            f"e.g. {', '.join(missing[:3])}")

    if cfg.dry_run:
        say("\nDry run — stopping here.")
        return

    all_fails = {}
    for L in cfg.primer_list:
        f = run_locus(L, cfg.samples, cfg)
        if f:
            all_fails[L["locus"]] = f

    say("\n" + "=" * 70)
    say(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] All loci finished")
    say(f"  summaries : {cfg.base_path}/{{cutadapt,bbduk,bbmerge}}_<LOCUS>_summary.tsv")
    say(f"  commands  : {cfg.base_path}/commands/")
    say(f"  tool logs : {cfg.base_path}/logs/")
    if all_fails:
        say("  FAILURES (see per-tool logs):")
        for locus, stages in all_fails.items():
            for stage, samples in stages.items():
                say(f"    {locus} / {stage}: {len(samples)} sample(s): "
                    f"{', '.join(samples[:5])}{' ...' if len(samples) > 5 else ''}")
    say("=" * 70)

    if cfg.plot:
        run_plot(cfg)

    if all_fails:
        sys.exit(1)


if __name__ == "__main__":
    main()

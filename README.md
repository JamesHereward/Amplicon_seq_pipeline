# Amplicon_seq_pipeline

Processes paired-end Illumina amplicon data for multiple loci on a local machine (Linux or WSL), no HPC scheduler needed.

For each locus in `primers.txt`, every sample in `sequences.txt` is run through:

1. **cutadapt**: removes the locus primers and discards read pairs without them
2. **bbduk**: trims adapters and quality-filters
3. **bbmerge**: merges overlapping pairs into single amplicon reads (input for DADA2 and similar)

The script gives the same outputs as the PBS version (`submit_amplicon_pipeline.sh` + `amplicon_pipeline.pbs`), so the downstream plotting script works on either.

---

## Requirements

| Tool | Notes |
|------|-------|
| Python ≥ 3.9 | standard library only |
| [cutadapt](https://cutadapt.readthedocs.io/) | must be on `PATH` (`pip install cutadapt`) |
| [BBTools](https://sourceforge.net/projects/bbmap/) | `bbduk.sh`, `bbmerge.sh` and `resources/adapters.fa`; point to them with `--bbmap` |
| Java | on `PATH`, needed by BBTools |
| pandas, matplotlib | only for `--plot` (`plot_pipeline_summary.py`) |

---

## Quick start

```bash
cd /mnt/c/FB                                   # project folder
python3 amplicon_pipeline_local.py --dry-run   # check inputs, tools and settings, then stop
python3 amplicon_pipeline_local.py -j 4 --plot # full run + summary plots
```

Project layout expected by the defaults:

```
project/
├── primers.txt
├── sequences.txt            ← here, or inside RAW/
├── amplicon_pipeline_local.py
├── plot_pipeline_summary.py ← optional, for --plot
└── RAW/                     ← or raw/
    ├── Sample001_1.fq.gz
    ├── Sample001_2.fq.gz
    └── ...
```

---

## Input files

### `primers.txt`

One locus per line, whitespace-delimited; lines starting with `#` are ignored.

```
# locus      fwd_name  fwd_seq                   rev_name  rev_seq
EPSPS2_4A    E24AF     TGTGGTGGTGTATTTCCCGT      E24AR     CAGAACATGCAACAAACAGAAAGG
COI          mlCOIintF GGWACWGGWTGAACWGTWTAYCCYCC jgHCO2198 TAIACYTCIGGRTGICCRAARAAYCA
```

If a primer has degenerate IUPAC bases (`R Y S W K M B D H V N I`), cutadapt runs with `--match-read-wildcards` for that locus automatically.

### `sequences.txt`

One read filename per line. R1 names, R2 names or a mix all work, because the read suffix is stripped to get the sample name.

```
Sample001_1.fq.gz
Sample001_2.fq.gz
Sample002_1.fq.gz
```

The read-naming convention is taken from the first entry. Supported suffixes:

| Convention | Example |
|------------|---------|
| bcl2fastq default | `Sample_R1_001.fastq.gz` |
| R1/R2 | `Sample_R1.fastq.gz`, `Sample_R1.fq.gz` |
| 1/2 | `Sample_1.fastq.gz`, `Sample_1.fq.gz` |
| bcl2fastq + fq.gz | `Sample_R1_001.fq.gz` |

Without a `sequences.txt`, the sample list is built from the R1 files in the raw directory. Windows (CRLF) line endings are fine in both input files.

---

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-b, --base-path DIR` | current directory | Root output directory |
| `-i, --in-path DIR` | `<base>/RAW`, else `<base>/raw` | Raw reads directory |
| `-s, --sequences FILE` | `./sequences.txt`, else `<in_path>/sequences.txt`, else built from R1 files | Sample list |
| `-p, --primers FILE` | `./primers.txt` | Primer definitions |
| `--bbmap PATH` | `/mnt/c/LP/bbmap/bbmap.sh` | `bbmap.sh` or the bbmap directory |
| `-t, --threads N` | all cores | Total threads |
| `-j, --jobs N` | 1 | Samples processed in parallel (threads are split between them) |
| `--mem SIZE` | BBTools auto (`2g` if `-j > 1`) | Java heap for bbduk/bbmerge |
| `-e, --error-rate X` | 0.1 | cutadapt `-e` (proportion of primer length) |
| `-O, --overlap N` | 3 | cutadapt `--overlap` minimum primer overlap (bp) |
| `-r, --relaxed` | off | Preset `-e 0.2 -O 10`, writes to `<base>/relaxed/` |
| `-l, --loci L1 L2 …` | all | Only run these loci |
| `--resume` | off | Skip sample/stage steps that already completed |
| `--plot` | off | Run `plot_pipeline_summary.py` when finished |
| `-m, --metadata FILE` | none | Metadata CSV passed to the plot script |
| `--dry-run` | off | Validate and print settings only |

### Examples

```bash
# Explicit paths
python3 amplicon_pipeline_local.py -b /mnt/c/FB -i /mnt/c/FB/RAW \
    -s /mnt/c/FB/RAW/sequences.txt -p /mnt/c/FB/primers.txt \
    --bbmap /mnt/c/LP/bbmap/bbmap.sh

# Re-run two loci with relaxed primer matching
python3 amplicon_pipeline_local.py -r -l EPSPS1_3B EPSPS1_7A

# Pick up where an interrupted run stopped
python3 amplicon_pipeline_local.py -j 4 --resume
```

### Performance

Amplicon files are small, so runtime is mostly tool start-up (BBTools launches a JVM per sample). Running several samples at once with `-j` is the main speed-up: e.g. `-t 16 -j 4` runs 4 samples at a time with 4 threads each.

### cutadapt error rate

`-e` is a proportion of primer length, not a percentage. For a 20 bp primer, `0.1` allows 2 mismatches and `0.2` allows 4. When raising `-e`, also raise `-O` to avoid spurious short matches. `-r` does both.

---

## Outputs

Everything is written under `<base_path>/`, with one set of folders per locus:

```
<base_path>/
├── <LOCUS>_cutadapt/            primer-trimmed reads + cutadapt JSON reports
├── <LOCUS>_bbduk/               adapter-trimmed, quality-filtered reads
├── <LOCUS>_merged/              <sample>_merged.fq.gz (+ unmerged R1/R2)
├── <LOCUS>_merge_hist/          per-sample insert-size histograms
├── cutadapt_<LOCUS>_summary.tsv
├── bbduk_<LOCUS>_summary.tsv
├── bbmerge_<LOCUS>_summary.tsv
├── commands/                    exact commands run, per tool and locus
└── logs/
    ├── amplicon_local_<timestamp>.log   full run log
    └── <tool>_<LOCUS>_<sample>.log      per-tool, per-sample logs
```

Summary TSV columns:

| File | Columns |
|------|---------|
| `cutadapt_<LOCUS>_summary.tsv` | `Sample  Input_Pairs  Pairs_Written  Pct_Written` |
| `bbduk_<LOCUS>_summary.tsv` | `Sample  Input_Reads  Result_Reads  Pct_Remaining` |
| `bbmerge_<LOCUS>_summary.tsv` | `Sample  Pairs  Joined  Pct_Joined` |

With `--plot`, `plot_pipeline_summary.py` adds combined summary TSVs, `pipeline_summary.png`, `cutadapt_summary.png`, `bbduk_bbmerge_summary.png`, per-locus insert-size histograms, and an interactive `pipeline_summary.html`.

---

## Behaviour notes

- **Missing raw files**: the sample is recorded as `NA` and skipped, and the run continues.
- **Samples with no primer match** (blanks, controls): the empty output skips bbduk/bbmerge cleanly and is recorded as `0`.
- **Tool failures**: recorded as `NA`, listed at the end of the run, and the script exits with code 1; other samples still run.
- **Plotting**: the plot script stays separate, and `--plot` calls it. It is looked for next to `amplicon_pipeline_local.py`, then in the base path. Old combined summary TSVs are deleted first so plots always reflect the current run.
- **Re-runs**: re-running without `--resume` overwrites previous outputs for the selected loci.

---

## Running on a PBS cluster

The original HPC version submits one PBS job per locus:

```bash
bash submit_amplicon_pipeline.sh -b /path/to/project -i /path/to/raw
```

It produces the same outputs and summary TSVs as the local script.

#!/bin/bash
# submit_amplicon_pipeline.sh
# ---------------------------
# Reads primers.txt and submits one PBS job per locus.
# Each job runs the full cutadapt → bbduk → bbmerge pipeline for that locus.
#
# Usage:
#   bash submit_amplicon_pipeline.sh [options]
#
# Options:
#   -b <base_path>    Root output directory  [default: current directory]
#   -i <in_path>      Raw reads directory    [default: <base_path>/raw]
#   -s <sequences>    sequences.txt path     [default: ./sequences.txt]
#   -p <primers>      primers.txt path       [default: ./primers.txt]
#   -w <walltime>     Walltime hh:mm:ss      [default: 24:00:00]
#   -m <mem>          Memory, e.g. 32gb      [default: 32gb]
#   -n <ncpus>        CPUs per job           [default: 40]
#   -e <error_rate>   cutadapt primer error rate [default: 0.1]
#                       e.g. -e 0.2  to allow up to ~20% mismatches
#   -O <overlap>      cutadapt minimum primer overlap in bp [default: 10]
#                       Raise together with -e to avoid spurious matches.
#                       e.g. -e 0.2 -O 10  (relaxed but still strict on overlap)
#   -r                Relaxed primer matching preset: -e 0.2 -O 10
#                       Shorthand for exploring loci with poor primer binding.
#                       Writes to <base_path>/relaxed/ to keep runs separate.
#
# primers.txt format (whitespace-delimited, one locus per line):
#   locus_name  fwd_name  fwd_seq  rev_name  rev_seq
# ---------------------------------------------------------------------------

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
BASE_PATH="$(pwd)"
IN_PATH=""
SEQ_FILE="sequences.txt"
PRIMERS_FILE="primers.txt"
WALLTIME="24:00:00"
MEM="32gb"
NCPUS=40
CA_ERROR_RATE="0.1"   # cutadapt -e
CA_OVERLAP="3"        # cutadapt --overlap (default matches cutadapt's own default)
RELAXED=0             # set by -r flag

# ── Argument parsing ─────────────────────────────────────────────────────────
while getopts "b:i:s:p:w:m:n:e:O:r" opt; do
    case $opt in
        b) BASE_PATH="$OPTARG" ;;
        i) IN_PATH="$OPTARG" ;;
        s) SEQ_FILE="$OPTARG" ;;
        p) PRIMERS_FILE="$OPTARG" ;;
        w) WALLTIME="$OPTARG" ;;
        m) MEM="$OPTARG" ;;
        n) NCPUS="$OPTARG" ;;
        e) CA_ERROR_RATE="$OPTARG" ;;
        O) CA_OVERLAP="$OPTARG" ;;
        r) RELAXED=1 ;;
        *) echo "Unknown option: -$opt"; exit 1 ;;
    esac
done

# Apply relaxed preset (can be overridden further by explicit -e/-O after -r)
if [[ "${RELAXED}" == "1" ]]; then
    CA_ERROR_RATE="${CA_ERROR_RATE:-0.2}"
    CA_OVERLAP="${CA_OVERLAP:-10}"
    # Only apply defaults if user hasn't already set them via -e/-O
    [[ "${CA_ERROR_RATE}" == "0.1" ]] && CA_ERROR_RATE="0.2"
    [[ "${CA_OVERLAP}"    == "3"   ]] && CA_OVERLAP="10"
    # Redirect output under a relaxed/ subdirectory so it doesn't clobber the
    # strict run.  The user can override -b explicitly after -r if desired.
    BASE_PATH="${BASE_PATH}/relaxed"
fi

[[ -z "${IN_PATH}" ]] && IN_PATH="${BASE_PATH%/relaxed}/raw"  # raw reads are always in the original base

BASE_PATH="$(realpath -m "${BASE_PATH}")"
IN_PATH="$(realpath "${IN_PATH}")"
SEQ_FILE="$(realpath "${SEQ_FILE}")"
PRIMERS_FILE="$(realpath "${PRIMERS_FILE}")"
PBS_SCRIPT="$(realpath "$(dirname "$0")/amplicon_pipeline.pbs")"

mkdir -p "${BASE_PATH}"

# ── Validate inputs ───────────────────────────────────────────────────────────
[[ ! -f "${SEQ_FILE}" ]]     && { echo "ERROR: sequences file not found: ${SEQ_FILE}";  exit 1; }
[[ ! -f "${PRIMERS_FILE}" ]] && { echo "ERROR: primers file not found: ${PRIMERS_FILE}"; exit 1; }
[[ ! -f "${PBS_SCRIPT}" ]]   && { echo "ERROR: PBS script not found: ${PBS_SCRIPT}";    exit 1; }
[[ ! -d "${IN_PATH}" ]]      && { echo "ERROR: input directory not found: ${IN_PATH}";  exit 1; }

LOG_DIR="${BASE_PATH}/logs"
mkdir -p "${LOG_DIR}"

# ── Validate error rate ───────────────────────────────────────────────────────
if ! [[ "${CA_ERROR_RATE}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "ERROR: -e must be a number, e.g. 0.1 or 0.2 (got: ${CA_ERROR_RATE})"
    exit 1
fi
if ! [[ "${CA_OVERLAP}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: -O must be a positive integer (got: ${CA_OVERLAP})"
    exit 1
fi

# Warn if error rate is very high (above 0.3 is almost certainly a mistake)
if awk "BEGIN{exit !(${CA_ERROR_RATE} > 0.3)}"; then
    echo "WARNING: error rate ${CA_ERROR_RATE} is very high (>0.3)."
    echo "  cutadapt error rate is a proportion of primer length, not a percentage."
    echo "  For a 20 bp primer: 0.1 → 2 mismatches, 0.2 → 4, 0.3 → 6."
    echo "  Continuing, but check your results carefully."
fi

# ── IUPAC degenerate base detection ──────────────────────────────────────────
is_degenerate() {
    echo "$1" | grep -qiE '[RYSWKMBDHVNI]'
}

# ── Detect read format from sequences.txt ────────────────────────────────────
detect_read_format_from_seqfile() {
    local seqfile="$1"
    local first_name
    first_name=$(grep -v '^\s*$' "${seqfile}" | head -1)

    local patterns=(
        "_R1_001.fastq.gz"
        "_R1.fastq.gz"
        "_1.fastq.gz"
        "_R1_001.fq.gz"
        "_R1.fq.gz"
        "_1.fq.gz"
    )
    for pat in "${patterns[@]}"; do
        if [[ "${first_name}" == *"${pat}" ]]; then
            echo "${pat}"
            return 0
        fi
    done
    echo ""
    return 1
}

READ_FMT_R1=$(detect_read_format_from_seqfile "${SEQ_FILE}")
if [[ -z "${READ_FMT_R1}" ]]; then
    echo "ERROR: Could not detect read format from first entry in ${SEQ_FILE}"
    echo "  First entry: $(grep -v '^\s*$' "${SEQ_FILE}" | head -1)"
    echo "  Expected suffix like _R1.fq.gz, _R1_001.fastq.gz, etc."
    exit 1
fi

READ_FMT_R2="${READ_FMT_R1/_R1/_R2}"
READ_FMT_R2="${READ_FMT_R2/_1./_2.}"
READ_EXT="${READ_FMT_R1##*_R1}"
READ_EXT="${READ_EXT##*_1}"

# ── Summary ───────────────────────────────────────────────────────────────────
echo "Submitting amplicon pipeline jobs"
echo "  base_path   : ${BASE_PATH}"
echo "  in_path     : ${IN_PATH}"
echo "  seq_file    : ${SEQ_FILE}"
echo "  primers     : ${PRIMERS_FILE}"
echo "  resources   : walltime=${WALLTIME}, mem=${MEM}, ncpus=${NCPUS}"
echo "  cutadapt -e : ${CA_ERROR_RATE}  ($(awk "BEGIN{printf \"%.0f\", ${CA_ERROR_RATE}*100}")% error rate)"
echo "  cutadapt -O : ${CA_OVERLAP} bp minimum overlap"
echo "  read format : R1=${READ_FMT_R1}  R2=${READ_FMT_R2}"
if [[ "${RELAXED}" == "1" ]]; then
    echo "  *** RELAXED PRESET ACTIVE — outputs in ${BASE_PATH} ***"
fi
echo ""

# ── Read primers.txt and submit one job per locus ─────────────────────────────
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "${line// }" ]] && continue
    [[ "${line}" =~ ^# ]]  && continue

    read -r LOCUS FWD_NAME FWD_SEQ REV_NAME REV_SEQ <<< "${line}"

    WILDCARD=0
    ( is_degenerate "${FWD_SEQ}" || is_degenerate "${REV_SEQ}" ) && WILDCARD=1

    echo "Submitting: ${LOCUS}"
    echo "  fwd: ${FWD_NAME} = ${FWD_SEQ}"
    echo "  rev: ${REV_NAME} = ${REV_SEQ}"
    echo "  wildcard: ${WILDCARD}"

    JOBLOG="${LOG_DIR}/amplicon_${LOCUS}.log"

    JOB_ID=$(qsub \
        -N "amplicon_${LOCUS}" \
        -l "walltime=${WALLTIME},mem=${MEM},ncpus=${NCPUS}" \
        -j oe \
        -o "${JOBLOG}" \
        -v "BASE_PATH=${BASE_PATH},LOCUS=${LOCUS},FWD_SEQ=${FWD_SEQ},REV_SEQ=${REV_SEQ},WILDCARD=${WILDCARD},IN_PATH=${IN_PATH},SEQ_FILE=${SEQ_FILE},READ_FMT_R1=${READ_FMT_R1},READ_FMT_R2=${READ_FMT_R2},READ_EXT=${READ_EXT},CA_ERROR_RATE=${CA_ERROR_RATE},CA_OVERLAP=${CA_OVERLAP}" \
        "${PBS_SCRIPT}")

    echo "  → submitted job ${JOB_ID}, log: ${JOBLOG}"
    echo ""

done < "${PRIMERS_FILE}"

echo "All jobs submitted. Monitor with: qstat -u \$USER"
if [[ "${RELAXED}" == "1" ]]; then
    echo ""
    echo "To compare results with your original run, point plot_pipeline_summary.py"
    echo "at both directories and check cutadapt Pairs_Written columns."
fi

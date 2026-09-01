#!/bin/bash
# Verify all CellBender sample directories have the complete expected output set.
# Usage: ./verify_cellbender.sh
# Exit 0 = all complete; exit 1 = one or more incomplete.

ROOT=/lus/lfs1aip2/projects/u6pl/ref_data/TAURUS_GSE282122/raw_processed_data

# Files every completed sample must have (10 files + slog dir checked separately)
EXPECTED=(
    rbg_output.log
    rbg_output.pdf
    rbg_output_cell_barcodes.csv
    rbg_output_filtered.h5
    rbg_output_metrics.csv
    rbg_output_posterior.h5
    rbg_output_report.html
)

total=0; ok=0; bad=0

for dir in "$ROOT"/*/; do
    [[ -d "$dir" ]] || continue
    sample=$(basename "$dir")
    ((total++))
    missing=()

    for f in "${EXPECTED[@]}"; do
        [[ -s "$dir/$f" ]] || missing+=("$f")   # -s: exists AND non-empty
    done
    [[ -d "$dir/slog" ]] || missing+=("slog/")

    if ((${#missing[@]} == 0)); then
        ((ok++))
    else
        ((bad++))
        printf 'INCOMPLETE  %-16s  missing: %s\n' "$sample" "${missing[*]}"
    fi
done

echo "--------------------------------------------------"
printf 'Total: %d   Complete: %d   Incomplete: %d\n' "$total" "$ok" "$bad"

((bad == 0)) && { echo "ALL COMPLETE"; exit 0; } || exit 1

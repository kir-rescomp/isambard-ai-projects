#!/bin/bash -e

#SBATCH --job-name  cellbender-taurus
#SBATCH --time      24:00:00
#SBATCH --partition workq
#SBATCH --nodes     1
#SBATCH --gpus      4
#SBATCH --output    slog/%j.out

source /lus/lfs1aip2/projects/u6pl/software/virtual_env/cellbender_py311/bin/activate

ROOT=/lus/lfs1aip2/projects/u6pl/ref_data/TAURUS_GSE282122/raw_processed_data

run_one() {
    local dir="$1"
    local gpu="$2"
    local sample; sample=$(basename "$dir")

    # Resume guard: skip samples already completed
    [[ -f "$dir/rbg_output.h5" ]] && { echo "SKIP $sample"; return; }

    mkdir -p "$dir/slog"
    CUDA_VISIBLE_DEVICES="$gpu" cellbender remove-background \
        --cuda \
        --input  "$dir/raw_feature_bc_matrix.h5" \
        --output "$dir/rbg_output.h5" \
        > "$dir/slog/${sample}.out" 2>&1 \
      && echo "OK   $sample" \
      || echo "FAIL $sample"
}
export -f run_one

# Feed sample dirs to 4 workers; GNU parallel slot {%} = 1..4 -> GPU 0..3
find "$ROOT" -mindepth 1 -maxdepth 1 -type d \
     -exec test -f '{}/raw_feature_bc_matrix.h5' ';' -print \
  | sort \
  | parallel --will-cite -j4 'run_one {} $(({%} - 1))'

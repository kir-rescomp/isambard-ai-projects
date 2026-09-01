#!/bin/bash
# Remove CellBender ckpt.tar.gz checkpoint files before backup.
# Dry-run by default. Pass --delete to actually remove.
# Usage:
#   ./clean_checkpoints.sh            # list what WOULD be deleted
#   ./clean_checkpoints.sh --delete   # actually delete

ROOT=/lus/lfs1aip2/projects/u6pl/ref_data/TAURUS_GSE282122/raw_processed_data

mode="dryrun"
[[ "$1" == "--delete" ]] && mode="delete"

count=0; bytes=0
while IFS= read -r f; do
    ((count++))
    sz=$(stat -c '%s' "$f" 2>/dev/null || echo 0)
    ((bytes+=sz))
    if [[ "$mode" == "delete" ]]; then
        rm -f "$f" && echo "DELETED  $f"
    else
        echo "WOULD DELETE  $f"
    fi
done < <(find "$ROOT" -mindepth 2 -maxdepth 2 -type f -name 'ckpt.tar.gz')

echo "--------------------------------------------------"
human=$(numfmt --to=iec-i --suffix=B "$bytes" 2>/dev/null || echo "${bytes}B")
if [[ "$mode" == "delete" ]]; then
    printf 'Deleted %d checkpoint files, freed %s\n' "$count" "$human"
else
    printf 'Found %d checkpoint files totalling %s (dry-run, nothing deleted)\n' "$count" "$human"
    echo "Re-run with --delete to remove them."
fi

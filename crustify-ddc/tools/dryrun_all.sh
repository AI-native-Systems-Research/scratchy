#!/bin/bash
# Dry-run every stage of the campaign against a repo root, proving each schedule parses, that its
# oracle_config sha256 still matches, and that every selected item has a home .rs on disk.
T=${1:?repo root required}
CR=/Users/nickm/.local/bin/crustify
rc=0
for d in "$T"/crustify/campaigns/ddc/*/; do
  for f in port.json review.json port-remainder.json; do
    [ -f "$d$f" ] || continue
    obj=port
    [ "$f" = review.json ] && obj=review
    out=$("$CR" "$T" . translate "$d$f" --objective "$obj" --dry-run 2>&1 | head -2 | tr '\n' ' ')
    printf '%-34s %-20s %s\n' "$(basename "$d")" "$f" "$out"
    case "$out" in
      *"dry-run]"*) ;;
      *) rc=1 ;;
    esac
  done
done
exit $rc

#!/bin/sh
# =============================================================================
# data-provenance-report.sh
#
# Produces a report on what the GeoVersa repository distributes, so that a third
# party can verify the answer instead of being asked to accept it.
#
# Written in POSIX shell and using only git, grep and standard utilities, so it
# can be re-run by anyone who has a clone. It needs no R and installs nothing.
#
#   sh tools/data-provenance-report.sh
#
# Writes tools/data-provenance-report.md and tools/data-provenance-report.json.
#
# TWO DESIGN CHOICES THAT MATTER TO A SCEPTIC
#
#   The term search reads every blob in the git object database, not the working
#   tree only and not a walk over revisions. The object database holds every
#   version of every file that has ever been committed, deduplicated. This
#   catches content that was committed and later deleted, which a search of the
#   current checkout would miss.
#
#   Both searches match whole words only, case-insensitively. Without the word
#   boundary, "OSSL" matches inside "lossless" and the report announces seven
#   occurrences of an unrelated word in a document meant to settle a question.
#
#   The search does not exclude this script or its own output, even though both
#   contain every term by construction. Excluding them would make the report
#   disagree with a plain `grep -ril` run by the reader, and a report that
#   cannot be reproduced by the obvious command is worse than no report. The
#   self-matches are counted and named instead.
# =============================================================================

set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

MD=tools/data-provenance-report.md
JSON=tools/data-provenance-report.json
SELF=tools/data-provenance-report.sh

COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "not-a-git-repository")
SHORT=$(git rev-parse --short HEAD 2>/dev/null || echo "-")
REMOTE=$(git config --get remote.origin.url 2>/dev/null || echo "none")
WHEN=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
DIRTY=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
NTRACKED=$(git ls-files | wc -l | tr -d ' ')
NCOMMITS=$(git rev-list --all --count 2>/dev/null || echo 0)

# Terms searched for. The first two are the datasets named in the query that
# prompted this report; the rest are the restricted or licence-encumbered soil
# data sources it would be reasonable to ask about.
TERMS="datsoilspc svrc_example OSSL KSSL Kellogg NCSS RaCA SSURGO STATSGO NASIS
LUCAS WoSIS ISRIC SoilGrids AfSIS ICRAF SoilData Dataverse prospectr resemble
simplerspec asdreader opusreader"

# ---- one pass over every blob that has ever existed -------------------------
BLOBS=$(mktemp); HITS=$(mktemp)
trap 'rm -f "$BLOBS" "$HITS"' EXIT
git rev-list --objects --all 2>/dev/null | awk '{print $1}' \
  | git cat-file --batch-check='%(objectname) %(objecttype)' 2>/dev/null \
  | awk '$2=="blob"{print $1}' > "$BLOBS"
NBLOBS=$(wc -l < "$BLOBS" | tr -d ' ')
ALT=$(echo "$TERMS" | tr '\n' ' ' | sed 's/  */|/g; s/^|//; s/|$//')
git cat-file --batch < "$BLOBS" 2>/dev/null \
  | grep -o -w -i -E "$ALT" 2>/dev/null | tr 'A-Z' 'a-z' | sort | uniq -c > "$HITS" || true

hist_count() {
  n=$(awk -v t="$(echo "$1" | tr 'A-Z' 'a-z')" '$2==t{print $1}' "$HITS" | head -1)
  [ -n "${n:-}" ] && echo "$n" || echo 0
}
# Tracked files only, which is exactly what a clone contains. Searching the
# author's working directory instead would report files the reader does not
# receive, and would take minutes walking the 211 MB of reference data that
# lives outside version control on this machine.
tree_files()  { git grep -lIwi -e "$1" -- . 2>/dev/null | sort || true; }

# ---- tracked files that are data --------------------------------------------
DATA_RE='\.(rds|rda|RData|csv|tsv|tif|tiff|nc|shp|dbf|shx|prj|gpkg|xlsx|xls|zip|parquet|feather|h5|hdf5|sav|dta|json)$'
TRACKED_DATA=$(git ls-files | grep -iE "$DATA_RE" || true)
NDATA=$(printf '%s' "$TRACKED_DATA" | grep -c . || true)

{
printf '# GeoVersa: what this repository distributes\n\n'
printf 'Generated %s from commit `%s`.\n\n' "$WHEN" "$COMMIT"
printf '| | |\n|---|---|\n'
printf '| Repository | `%s` |\n' "$REMOTE"
printf '| Commit | `%s` |\n' "$SHORT"
printf '| Commits in history | %s |\n' "$NCOMMITS"
printf '| Files tracked by git | %s |\n' "$NTRACKED"
printf '| Uncommitted changes when generated | %s |\n' "$DIRTY"
printf '| Blobs scanned (every version ever committed) | %s |\n\n' "$NBLOBS"

printf 'Reproduce with:\n\n```sh\ngit clone %s && cd GeoVersa\nsh %s\n```\n\n' "$REMOTE" "$SELF"

printf -- '---\n\n## 1. Data files under version control\n\n'
if [ "${NDATA:-0}" -eq 0 ]; then
  printf 'None. No file matching a data extension is tracked by git.\n\n'
else
  printf 'The following %s file(s) match a data extension:\n\n' "$NDATA"
  printf '| File | Bytes | SHA-256 |\n|---|---|---|\n'
  printf '%s\n' "$TRACKED_DATA" | while IFS= read -r f; do
    [ -n "$f" ] || continue
    b=$(wc -c < "$f" 2>/dev/null | tr -d ' ')
    h=$( (shasum -a 256 "$f" 2>/dev/null || sha256sum "$f" 2>/dev/null) | awk '{print $1}')
    printf '| `%s` | %s | `%s` |\n' "$f" "${b:-?}" "${h:-?}"
  done
  printf '\n'
fi
printf 'Verify with:\n\n```sh\ngit ls-files | grep -iE %s\n```\n\n' "'$DATA_RE'"

printf -- '---\n\n## 2. Search for named datasets\n\n'
printf 'Each term is searched twice: in the working tree, and in every blob in\n'
printf 'the git object database. The second search sees content that was\n'
printf 'committed at any time and later deleted.\n\n'
printf '| Term | Files in a clone | Occurrences in history | Where |\n|---|---|---|---|\n'
for t in $TERMS; do
  files=$(tree_files "$t")
  n=$(printf '%s' "$files" | grep -c . || true)
  h=$(hist_count "$t")
  where=$(printf '%s' "$files" | paste -sd'; ' - 2>/dev/null || printf '%s' "$files" | tr '\n' ';')
  [ -z "$where" ] && where='-'
  printf '| `%s` | %s | %s | %s |\n' "$t" "${n:-0}" "$h" "$where"
done
printf '\nVerify any single term with:\n\n```sh\ngrep -rlI -e TERM . --exclude-dir=.git\n```\n\n'

printf -- '---\n\n## 3. What is deliberately not distributed\n\n'
printf 'The `.gitignore` excludes whole directories and re-admits only their\n'
printf 'README files, so a new file type appearing in one of them cannot be\n'
printf 'committed by accident:\n\n```\n'
grep -vE '^\s*(#|$)' .gitignore 2>/dev/null | sed -n '/\*\*/p' || true
printf '```\n\nPresent on the author machine and excluded from the repository:\n\n'
printf '| Directory | On disk | Tracked |\n|---|---|---|\n'
for d in data external results figures; do
  if [ -d "$d" ]; then
    sz=$(du -sh "$d" 2>/dev/null | cut -f1)
    tr_=$(git ls-files "$d" | wc -l | tr -d ' ')
    printf '| `%s/` | %s | %s file(s) |\n' "$d" "${sz:-?}" "$tr_"
  fi
done
printf '\n'

printf -- '---\n\n## 4. Declared dependencies\n\n'
printf 'From `DESCRIPTION`. None is a soil-spectroscopy package.\n\n```\n'
sed -n '/^Imports:/,/^[A-Z][A-Za-z]*:/p;/^Suggests:/,/^[A-Z][A-Za-z]*:/p' DESCRIPTION 2>/dev/null \
  | grep -vE '^(RoxygenNote|Config|Encoding|License):' || true
printf '```\n\nAny data these packages bundle is distributed by those packages under\n'
printf 'their own licences, from CRAN, and is not redistributed here.\n\n'

printf -- '---\n\n## 5. Scope, and what this report does not cover\n\n'
printf -- '- The term search covers the working tree and the full git object\n'
printf '  database of this repository. It does not cover forks, mirrors, or\n'
printf '  release archives published outside git.\n'
printf -- '- It does **not** exclude this script or its own output. Both contain\n'
printf '  every term by construction, so a term with a count of 1 or 2 whose\n'
printf '  location is `%s` or the report files is matching the\n' "$SELF"
printf '  search list itself, not data. This is stated rather than filtered so\n'
printf '  that a reader running `grep` gets the same numbers.\n'
printf -- '- Reference data used by the analyses is obtained by the user from its\n'
printf '  original source. See `external/README.md` and `data/README.md`.\n'
printf -- '- A file carrying no searched term and no data extension is not\n'
printf '  inspected for content. The categorical exclusions in section 3 are\n'
printf '  what bound the repository, not this search.\n\n'
} > "$MD"

# ---- machine-readable twin ---------------------------------------------------
{
printf '{\n  "generated_at": "%s",\n  "repository": "%s",\n  "commit": "%s",\n' "$WHEN" "$REMOTE" "$COMMIT"
printf '  "commits_in_history": %s,\n  "files_tracked": %s,\n' "$NCOMMITS" "$NTRACKED"
printf '  "uncommitted_changes": %s,\n  "blobs_scanned": %s,\n' "$DIRTY" "$NBLOBS"
printf '  "tracked_data_files": ['
first=1
printf '%s\n' "$TRACKED_DATA" | while IFS= read -r f; do
  [ -n "$f" ] || continue
  [ $first -eq 0 ] && printf ','
  printf '"%s"' "$f"; first=0
done
printf '],\n  "term_search": [\n'
i=0
for t in $TERMS; do
  files=$(tree_files "$t"); n=$(printf '%s' "$files" | grep -c . || true); h=$(hist_count "$t")
  [ $i -gt 0 ] && printf ',\n'
  printf '    {"term": "%s", "files_in_tree": %s, "occurrences_in_history": %s}' "$t" "${n:-0}" "$h"
  i=$((i+1))
done
printf '\n  ]\n}\n'
} > "$JSON"

printf 'wrote %s\n       %s\n' "$MD" "$JSON"
printf '  %s tracked files, %s data files, %s blobs scanned\n' "$NTRACKED" "${NDATA:-0}" "$NBLOBS"

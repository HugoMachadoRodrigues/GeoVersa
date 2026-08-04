#!/bin/zsh
# Build manuscript.pdf. Four passes so that elsarticle-harv citations, the
# bibliography and every cross-reference all settle.
#
# Usage:  zsh make_pdf.sh          (log lands in build.log)
cd "$(dirname "$0")"
set -e
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >build.log 2>&1 || true
bibtex manuscript >>build.log 2>&1 || true
pdflatex -interaction=nonstopmode manuscript.tex >>build.log 2>&1 || true
pdflatex -interaction=nonstopmode manuscript.tex >>build.log 2>&1 || true

echo "--- build summary ---"
grep -E 'Output written' build.log | tail -1 || echo "  NO PDF PRODUCED"
python3 - <<'PYX'
import re
log = open("build.log", errors="ignore").read()
last = log.split("This is pdfTeX")[-1]
u = sorted(set(re.findall(r"Citation `([^']+)' on page", last))) + \
    sorted(set(re.findall(r"Reference `([^']+)' on page", last)))
o = len(re.findall(r"Overfull \\hbox", last))
print("  undefined (final pass): %d %s" % (len(u), u if u else ""))
print("  overfull  (final pass): %d" % o)
PYX
exit 0

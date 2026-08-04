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
grep -E 'Output written' build.log || echo "  NO PDF PRODUCED"
u=$(grep -c 'undefined' build.log || true)
echo "  undefined references/citations: ${u}"
[ "$u" -gt 0 ] && grep 'undefined' build.log | head -10
o=$(grep -c 'Overfull \\hbox' build.log || true)
echo "  overfull hboxes: ${o}"
exit 0

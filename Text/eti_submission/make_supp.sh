#!/bin/zsh
# Build supplementary.pdf. Same four-pass sequence as the main manuscript.
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode supplementary.tex >supp.log 2>&1 || true
bibtex supplementary >>supp.log 2>&1 || true
pdflatex -interaction=nonstopmode supplementary.tex >>supp.log 2>&1 || true
pdflatex -interaction=nonstopmode supplementary.tex >>supp.log 2>&1 || true
echo "--- build summary ---"
grep -E 'Output written' supp.log | tail -1 || echo "  NO PDF PRODUCED"
python3 - <<'PYX'
import re
log = open("supp.log", errors="ignore").read()
last = log.split("This is pdfTeX")[-1]
u = sorted(set(re.findall(r"Citation `([^']+)' on page", last))) + \
    sorted(set(re.findall(r"Reference `([^']+)' on page", last)))
o = len(re.findall(r"Overfull \\hbox", last))
print("  undefined (final pass): %d %s" % (len(u), u if u else ""))
print("  overfull  (final pass): %d" % o)
PYX
exit 0

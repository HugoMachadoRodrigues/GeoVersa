#!/bin/zsh
cd "$(dirname "$0")"
pandoc manuscript.tex \
  --bibliography=references.bib --citeproc \
  --resource-path=.:figures \
  -o manuscript.docx

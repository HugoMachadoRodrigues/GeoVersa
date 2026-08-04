# Submission package: Ecological Modelling

Wadoux et al. (2021), the paper this one extends, appeared in this journal, so
the argument lands where the conversation already is.

Build:

    zsh make_pdf.sh     # manuscript.pdf, reports undefined refs and overfull lines
    zsh make_docx.sh    # manuscript.docx, from the same source

## What to upload

| File | Role |
|---|---|
| `manuscript.tex` / `manuscript.docx` | the manuscript, 26 pages |
| `references.bib` | 26 references |
| `figures/` | four manuscript figures |
| `highlights.md` | required; 5 bullets, each within 85 characters |
| `figures/graphical_abstract.png` | encouraged; 3984 x 1593 px |

## Journal rules this package is built against

* Abstract at most 250 words. Currently 248.
* Keywords 1 to 7. Currently 6.
* Highlights required, 3 to 5 bullets, at most 85 characters each.
* Graphical abstract encouraged, not required.
* No word limit and no cap on figures and tables for a regular paper, which is
  why the procedure appendix and all eight tables sit in the main text. An
  earlier pass cut them to fit a different journal; that is not needed here.
* Author-date citations, `elsarticle-harv`.

## Reproducible figure scripts, in the repository root `code/`

* `figures_schematic.R` the pipeline diagram, drawn from the correction library
  actually implemented in `R/dbtarf.R`
* `figures_protocols.R` the estimand comparison, read from the result CSVs
* `maps_compute.R` and `maps_plot.R` whole-grid prediction and the map panel
* `figures_graphical_abstract.R` the graphical abstract, numbers taken from the
  coverage table
* `tables_manuscript.R` every table, generated from the result CSVs so no number
  in the paper can drift from the run that produced it

## Left to the corresponding author

1. The data statement says the code is "archived with a persistent identifier on
   release". Mint the Zenodo DOI before submitting, or soften that sentence.
2. The declaration of generative AI describes the tools actually used. Review it
   and decide what you are willing to sign.
3. Author order and the CRediT roles for both authors are set as agreed; confirm
   with the co-author before submission, since this journal does not permit
   authorship changes after acceptance.

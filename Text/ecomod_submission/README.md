# Submission package: Ecological Modelling

Wadoux et al. (2021), the paper this one extends, appeared in this journal, so
the argument lands where the conversation already is.

Build:

    zsh make_pdf.sh     # manuscript.pdf, reports undefined refs and overfull lines
    zsh make_docx.sh    # manuscript.docx, from the same source

## What to upload

| File | Role |
|---|---|
| `manuscript.tex` / `manuscript.docx` | the manuscript, 33 pages |
| `references.bib` | 30 references |
| `figures/` | the four manuscript figures, all script-generated |
| `tables/` | nine generated tables the manuscript `\input`s |
| `highlights.md` | required; 5 bullets, each within 85 characters |
| `figures/graphical_abstract.png` | encouraged; 3984 x 1592 px |
| `elsevier-harv-expanded.csl` | needed by `make_docx.sh`, not uploaded |

The manuscript carries ten tables: the nine generated ones plus a descriptive
table of the three sampling mechanisms, which contains no data and is written
inline.

## Journal rules this package is built against

* Abstract at most 250 words. Currently 223.
* Keywords 1 to 7. Currently 6.
* Highlights required, 3 to 5 bullets, at most 85 characters each.
* Graphical abstract encouraged, not required.
* No word limit and no cap on figures and tables for a regular paper, which is
  why the procedure appendix and every table sit in the main text.
* Author-date citations, `elsarticle-harv`.

## Reproducible scripts, in the repository root `code/`

* `project_paths.R` resolves the repository, the benchmark data and the campaign
  outputs without any absolute path. Every other script bootstraps it.
* `tables_manuscript.R` all nine generated tables, built from the campaign CSVs
  so no number in the paper can drift from the run that produced it.
* `figures_schematic.R` the pipeline diagram, drawn from the correction library
  actually implemented in `R/dbtarf.R`
* `figures_protocols.R` the estimand comparison
* `figures_uncertainty.R` the interval-assessment figure, reading the same
  campaign CSVs as the coverage table so the two cannot disagree
* `figures_graphical_abstract.R` the graphical abstract, also from those CSVs
* `maps_compute.R` and `maps_plot.R` whole-grid prediction and the map panel
* `run_all_campaigns.sh` the campaign set; `freeze_check.sh` the pre-flight that
  gates it

Campaign outputs live outside the synced folder, resolved through
`GEOVERSA_RESULTS` or `~/geoversa_results`. Every results directory records the
git commit, whether the tree was dirty, the R version and a timestamp, and the
table generator refuses any arm whose config lacks a commit.

## Left to the corresponding author

1. **Author order** is Hugo Rodrigues then Golmar Golmohammadi, with Hugo
   Rodrigues as corresponding author. This journal does not permit authorship
   changes after acceptance, so confirm with the co-author before submitting.
2. **The generative-AI declaration** is the one section whose wording is still
   open. It currently describes the tools actually used.
3. **Sample-size sensitivity** covers n = 200 and n = 500 for the coverage
   result; an n = 800 pair is running and slots into `tables/sizes.tex` on its
   own when complete, since the generator includes a size only once its
   campaign holds 30 repetitions and records a git commit. Everything other
   than coverage still comes from n = 500, as Limitations says.

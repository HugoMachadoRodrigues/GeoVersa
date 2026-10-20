# Submission package: Ecological Modelling

Wadoux et al. (2021), the paper this one extends, appeared in this journal, so
the argument lands where the conversation already is.

Build:

    zsh make_pdf.sh     # manuscript.pdf, reports undefined refs and overfull lines
    zsh make_docx.sh    # manuscript.docx, from the same source

## What to upload

| File | Role |
|---|---|
| `manuscript.tex` / `manuscript.docx` | the manuscript, 42 pages |
| `references.bib` | 30 references |
| `figures/` | the five manuscript figures; not under version control, see below |
| `tables/` | ten generated tables the manuscript `\input`s |
| `highlights.md` | required; 5 bullets, each within 85 characters |
| `cover-letter.md` | one page; two slots to fill, editor name and date |
| `figures/graphical_abstract.png` | encouraged; 3984 x 1593 px |
| `elsevier-harv-expanded.csl` | needed by `make_docx.sh`, not uploaded |

The manuscript carries eleven tables: the ten generated ones plus a descriptive
table of the three sampling mechanisms, which contains no data and is written
inline.

## Journal rules this package is built against

* Abstract at most 250 words. Currently 231.
* Keywords 1 to 7. Currently 6.
* Highlights required, 3 to 5 bullets, at most 85 characters each.
* Graphical abstract encouraged, not required.
* No word limit and no cap on figures and tables for a regular paper, which is
  why the procedure appendix and every table sit in the main text.
* Author-date citations, `elsarticle-harv`.

## Reproducible scripts, in the repository root `code/`

* `project_paths.R` resolves the repository, the benchmark data and the campaign
  outputs without any absolute path. Every other script bootstraps it.
* `tables_manuscript.R` all ten generated tables, built from the campaign CSVs
  so no number in the paper can drift from the run that produced it.
* `figures_schematic.R` the pipeline diagram, drawn from the correction library
  actually implemented in `R/dbtarf.R`
* `figures_protocols.R` the estimand comparison
* `figures_uncertainty.R` the interval-assessment figure, reading the same
  campaign CSVs as the coverage table so the two cannot disagree
* `figures_distance.R` the error-against-distance figure, from the per-point
  dumps written by `run_point_dump.sh` with `DBT_DUMP_POINTS=1`
* `figures_graphical_abstract.R` the graphical abstract, from the per-point
  dumps and the sensitivity campaigns
* `maps_compute.R` and `maps_plot.R` whole-grid prediction and the map panel
* `run_all_campaigns.sh` the headline campaigns, `run_size_sweep.sh` the
  sensitivity sweep and `run_point_dump.sh` the per-point dumps;
  `freeze_check.sh` is the pre-flight that gates a launch and refuses a dirty
  working tree

Campaign outputs live outside the synced folder, resolved through
`GEOVERSA_RESULTS` or `~/geoversa_results`. Every results directory records the
git commit, whether the tree was dirty, the R version and a timestamp, and the
table generator refuses any arm whose config lacks a commit.

### Regenerating the figures and tables

The generators are under version control; their inputs are not. Of the five
manuscript figures, three are drawn from campaign outputs, one needs the
reference grid and about eighty minutes of prediction over 1.3 M cells, and one
runs on a bare clone. Every table is drawn from campaign outputs. Those outputs
run to 833 MB and are excluded from the repository along with the reference
data, so a clone does not rebuild the manuscript in one command. The chain is:

1. Obtain the benchmark materials as `external/README.md` describes. They are
   not redistributed here.
2. Run the campaigns: `code/run_all_campaigns.sh` for the three headline
   designs, `code/run_size_sweep.sh` for the size and geometry sweep, and
   `code/run_point_dump.sh` for the per-point records behind Figure 5. This is
   days of computation, not minutes.
3. Run `code/maps_compute.R` and `code/maps_plot.R` for the map panel, which
   fits on one calibration sample and predicts the whole grid.
4. Run `code/tables_manuscript.R` and the `code/figures_*.R` scripts, then
   `make_pdf.sh`.

`code/figures_schematic.R` is the exception: it draws from the correction
library in `R/dbtarf.R` alone and runs on a bare clone. `code/freeze_check.sh`
verifies on the author machine that every generator reproduces its output byte
for byte from the recorded campaigns, which is the check that the scripts and
the published numbers have not drifted apart.

## Left to the corresponding author

1. **Author order** is Hugo Rodrigues then Golmar Golmohammadi, with Hugo
   Rodrigues as corresponding author. This journal does not permit authorship
   changes after acceptance, so confirm with the co-author before submitting.
2. **The generative-AI declaration** is the one section whose wording is still
   open. It currently describes the tools actually used. The recommendation is
   to leave it until the journal is fixed, then match that journal's required
   form literally.
3. **The repository is public and archived on Zenodo from v0.5.1.** Zenodo's
   GitHub integration fetches the release archive without credentials, so the
   two v0.5.0 releases, published while the repository was private, failed
   with `"404 Not Found"`. Making the repository public does not re-trigger
   them: Zenodo archives only releases published after its webhook exists.
   v0.5.1 is the first release made with the repository public.

   The statement cites commit `5db0ec3`, where the reported results were
   produced, and v0.5.1, which contains it. If further commits land before
   submission, leave the citation at the commit that produced the results.
4. **Sample-size sensitivity** is reported in Section 7.6 and Table 8, from
   seven campaigns of 30 repetitions: simple random at n = 200, 500 and 800, and
   two clustered arms crossing at a shared 20 x 25 midpoint.

   Two traps are worth knowing about before extending it. `DBT_SAMPLE_SIZE` sets
   the calibration draw under the random and systematic designs but *not* under
   the clustered one, which takes its size from `DBT_N_PSU` times
   `DBT_M_PER_PSU`; a first attempt at the sweep set only the former and so
   shrank the validation sample while training on 500 regardless. And primary
   units are drawn with replacement from one hundred candidates, so the nominal
   unit count overstates the distinct one: 8, 20 and 32 realise 7.7, 18.2 and
   27.5.

   `tables_manuscript.R` reads `n_train` back from each campaign's diagnostics
   and drops any row whose calibration sample is not the size the row claims,
   requires a validation draw of 500 so the standard deviations are comparable,
   and deletes a previously written `tables/sizes.tex` rather than leaving a
   disqualified table on disk for the manuscript to include.

5. **The per-point dumps** behind Figure 5 are three campaigns of 10
   repetitions, run by `run_point_dump.sh`. They are the only outputs holding
   one row per validation point; every other campaign is aggregated per
   repetition, which is why the mechanism could not be examined before.

# GeoVersa: what this repository distributes

Generated 2026-08-28T20:13:38Z from commit `69ceb99e117441c6829e35ac0c81f10cbec1f7cd`.

| | |
|---|---|
| Repository | `https://github.com/HugoMachadoRodrigues/GeoVersa.git` |
| Commit | `69ceb99` |
| Commits in history | 124 |
| Files tracked by git | 104 |
| Uncommitted changes when generated | 3 |
| Blobs scanned (every version ever committed) | 442 |

Reproduce with:

```sh
git clone https://github.com/HugoMachadoRodrigues/GeoVersa.git && cd GeoVersa
sh tools/data-provenance-report.sh
```

---

## 1. Data files under version control

The following 3 file(s) match a data extension. The extension list is
deliberately broad and matches configuration and report files too; the
first line carrying text is printed so the reader can see what it holds.

| File | Lines | Bytes | First line | SHA-256 |
|---|---|---|---|---|
| `.zenodo.json` | 43 | 2159 | `"title": "GeoVersa: Design-Based Residual-Correction For` | `57ee07fcf3333004f185f8a7f18e189147fa274440bb3e6a819d320e65860553` |
| `docs/wadoux2021-reference/official_rdata_manifest.csv` | 4 | 907 | `"scenario","expected_rdata","full_path","exists","upstream` | `347a4d3b14de0c909f6e01a5ce1b71bbccc5b6356c8ea357a344f4bcff05aed9` |
| `tools/data-provenance-report.json` | 35 | 2137 | `"generated_at": "2026-08-28T20:13:14Z",` | `aaa13a6c01f4079cf562dd61b3af36dd78a8ec9ce2c9efba7ab92ee455b0a9e4` |

Verify with:

```sh
git ls-files | grep -iE '\.(rds|rda|RData|csv|tsv|tif|tiff|nc|shp|dbf|shx|prj|gpkg|xlsx|xls|zip|parquet|feather|h5|hdf5|sav|dta|json)$'
```

---

## 2. Search for named datasets

Each term is searched twice: in the working tree, and in every blob in
the git object database. The second search sees content that was
committed at any time and later deleted.

| Term | Files in a clone | Occurrences in history | Where |
|---|---|---|---|
| `datsoilspc` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `svrc_example` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `OSSL` | 2 | 6 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `KSSL` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `Kellogg` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `NCSS` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `RaCA` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `SSURGO` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `STATSGO` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `NASIS` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `LUCAS` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `WoSIS` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `ISRIC` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `SoilGrids` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `AfSIS` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `ICRAF` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `SoilData` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `Dataverse` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `prospectr` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `resemble` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `simplerspec` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `asdreader` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |
| `opusreader` | 2 | 4 | tools/data-provenance-report.json;tools/data-provenance-report.sh |

Verify any single term with:

```sh
grep -rlI -e TERM . --exclude-dir=.git
```

---

## 3. What is deliberately not distributed

The `.gitignore` excludes whole directories and re-admits only their
README files, so a new file type appearing in one of them cannot be
committed by accident:

```
**/.Rhistory
**/.RData
**/.Ruserdata
results/**
figures/**
data/**
external/**
Text/**
```

Present on the author machine and excluded from the repository:

| Directory | On disk | Tracked |
|---|---|---|
| `data/` | 1.3M | 1 file(s) |
| `external/` | 211M | 1 file(s) |
| `results/` | 720M | 1 file(s) |
| `figures/` | 4.0K | 1 file(s) |

---

## 4. Declared dependencies

From `DESCRIPTION`. None is a soil-spectroscopy package.

```
Imports:
    ranger,
    caret,
    stats,
    withr
Suggests:
    Cubist,
    gstat,
    sp,
    nnet,
    testthat (>= 3.0.0)
```

Any data these packages bundle is distributed by those packages under
their own licences, from CRAN, and is not redistributed here.

---

## 5. Scope, and what this report does not cover

- The term search covers the files a clone receives, read through
  `git grep`, and every blob in the object database, which includes
  content committed at any time and later deleted. It does not cover
  forks, mirrors, release archives published outside git, or untracked
  files that exist only on the author machine and are not distributed.
- It does **not** exclude this script or its own output. Both contain
  every term by construction, so a term with a count of 1 or 2 whose
  location is `tools/data-provenance-report.sh` or the report files is matching the
  search list itself, not data. This is stated rather than filtered so
  that a reader running `grep` gets the same numbers.
- Reference data used by the analyses is obtained by the user from its
  original source. See `external/README.md` and `data/README.md`.
- A file carrying no searched term and no data extension is not
  inspected for content. The categorical exclusions in section 3 are
  what bound the repository, not this search.


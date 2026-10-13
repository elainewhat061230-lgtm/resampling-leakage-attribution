# Licence for data outputs, figures, tables and documentation

Copyright (c) 2026 Yixuan Xiao

Except where stated otherwise below, the data outputs, figures, tables and documentation in this
repository are released under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

This covers in particular:

- the per-fold and per-cell outputs in `Experiment/raw/`, `Experiment/raw_diag/` and
  `Experiment/pilot_4state/`, the derived tables `Experiment/decomposed_cells.csv` and
  `Experiment/rq5_explanations.csv`, and the archived outputs in `Experiment/archive/`;
- the figures in `Experiment/figures/` (and the archived and pilot figures) and the tables in
  `Experiment/tables/` (and the archived and pilot tables);
- the results files, logs and documentation, including `README.md`,
  `Experiment/RESULTS_GROUND_TRUTH.md`, `Experiment/DEVIATIONS.md`,
  `Experiment/VERIFY_INDEPENDENT.md` and the preregistration files in `Ideation/`.

The source code (all `.py` files, including the archived code snapshot) is licensed separately
under the MIT licence; see `LICENSE`.

## Third-party datasets (not included)

The six OpenML datasets used in the study (OpenML IDs 1597, 1461, 1590, 42477, 31 and 37) are not
included in this repository. The code downloads them from OpenML at runtime, and each keeps its
own licence as stated on its OpenML page. The outputs in this repository contain evaluation
metrics, counts and diagnostics only; they contain no rows of these datasets.

## Journal corpus snapshot

`Experiment/corpus/ijdns_corpus_dedup_2026-08-30.json` is a snapshot of bibliographic metadata
of 983 articles published in the International Journal of Data and Network Science
(ISSN 2561-8156, publication years 2017 to 2026). The metadata were retrieved from Crossref and
DOAJ on 2026-08-30, merged by DOI and de-duplicated by title. Each record holds the DOI, title,
abstract, keywords, publication year, volume, issue, pages, page count, author names, number of
authors, citation and reference counts, publication type and DOI URL, plus fields added in this
project: study-type labels (`shapes`, `shape`), topic labels (`topics`, `topic`) and a citation
percentile (`cite_pct`). `Experiment/code/corpus_metrics.py` uses the snapshot to compute the
metric-reporting statistics in Section 7 of `Experiment/RESULTS_GROUND_TRUTH.md`.

The article metadata, including author names and abstracts, come from the publisher through
Crossref and DOAJ and remain subject to the terms of those sources and of the rights holders.
The CC BY 4.0 licence above applies only to this project's own contribution to the snapshot
(the compilation and the added fields), to the extent that contribution is protected.

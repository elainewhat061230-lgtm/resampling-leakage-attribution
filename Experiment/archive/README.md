# Archive: superseded designs and outputs

This folder keeps earlier outputs of the project for transparency. The results reported in the
paper come from the four-state design (`Experiment/code/states.py`, outputs in `Experiment/raw/`,
`Experiment/raw_diag/` and `Experiment/RESULTS_GROUND_TRUTH.md`). Nothing in this folder feeds
those results, with one exception: the supplementary controls described below, which
`Experiment/code/build_ground_truth.py` reads from `five_arm_design_2026-09-01/`.

## `five_arm_design_2026-09-01/`

Complete outputs of the earlier five-arm design (arms B, A, C, C2, D, E) in their final state of
2026-09-01, archived on 2026-09-25 when the four-state ordered decomposition replaced that design
(see `Experiment/DEVIATIONS.md`, entry D-011). The folder has its own README in Chinese.

| Path | Content |
|---|---|
| `raw/` | 2,620 per-cell CSV files (main 2,160, extreme 20, ladder 440), one row per fold and arm |
| `decomposed_cells.csv` | per-cell three-mechanism decomposition of the old design |
| `RESULTS_GROUND_TRUTH.md`, `figure_numbers.log`, `figures/`, `tables/` | results file, figure numbers, figures and tables of the old design |
| `rq5_explanations.csv`, `scaler_sensitivity.csv` | old explanation-agreement (RQ5) detail and old scaler sensitivity check |
| `code/` | snapshot of the scripts of the old design |
| `logs/` | run logs of the old main, extreme and ladder grids |

Supplementary controls read by the current code (Section 10 of `Experiment/RESULTS_GROUND_TRUTH.md`):

- `raw/main__*.csv`, rows with `arm == "B"`: old arm B, whose encoding and scaling were fitted on the
  full data, compared with CON, whose preprocessing is fitted on the training fold only (Section 10.1).
- `decomposed_cells.csv`: the old D minus C2 ablation, synthetic versus original minority rows
  (Section 10.2), and the additivity gap of the old paired-control design (Section 10.3).

## `superseded_outputs_2026-08-30_to_2026-09-01.zip`

Outputs superseded between 2026-08-30 and 2026-09-01. No current script reads them. Paths inside
the zip are relative to the repository root, so the files unpack under `Experiment/`. To inspect
them without mixing them into the working tree, unzip into a separate folder, for example:

```
unzip Experiment/archive/superseded_outputs_2026-08-30_to_2026-09-01.zip -d superseded
```

| Path inside the zip | Content |
|---|---|
| `Experiment/raw_v1_flawed_cap/` | 2,160 per-cell CSV files of the first main grid, run under the row-cap rule of preregistration v1, which distorted the class ratio of three datasets and inverted it for adult (DEVIATIONS D-001) |
| `Experiment/decomposed_cells_v1_flawed.csv`, `Experiment/RESULTS_GROUND_TRUTH_v1_flawed.md` | derived table and results file of that grid |
| `Experiment/raw_v2_pre_dualC/` | 720 CSV files for credit-g and pima before the "dual C" correction of the five-arm design (DEVIATIONS D-004): 719 main-grid cells plus `regen_test__credit-g__LogReg__SMOTE__seed00.csv`, which stands in for the missing `main__` file of that cell |
| `Experiment/decomposed_cells_v2_pre_dualC.csv`, `Experiment/RESULTS_GROUND_TRUTH_v2_pre_dualC.md` | derived table and results file before that correction |
| `Experiment/rq5_explanations_v1_mixedmethods.csv` | first RQ5 detail, which mixed importance methods across models (DEVIATIONS D-005) |
| `Experiment/SKELETON_VERIFICATION.md`, `Experiment/skeleton_check.csv` | skeleton verification of the five-arm design, run before the preregistration was frozen |
| `Experiment/main_grid_stdout.log`, `Experiment/main_grid_v2_stdout.log`, `Experiment/wait_and_build.log`, `Experiment/wait_and_build_v2.log`, `Experiment/extreme_ladder.log` | console logs of the v1 and v2 grid runs and of the result builds that followed them |

Absolute paths in the logs, here and in `five_arm_design_2026-09-01/logs/`, were replaced with
`<project-root>` or `<home>` before release. No other content was changed.

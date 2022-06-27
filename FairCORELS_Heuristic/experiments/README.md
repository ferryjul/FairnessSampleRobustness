Our experiments can be run on a computing grid using Slurm and MPI for parallelization (with the `mpi4py` library).

## Running experiments

Before running the experiments, it is necessary to unzip the `data.zip` folder, containing the datasets.
To launch all experiments: `./run_all.sh`

To launch experiments only for dataset `i`: `./run_all.sh i` (where `i` can be either `1`, `2`, `3` or `4` (`1`: Adult, `2`: COMPAS, `3`: Marketing, `4`: Default Credit)).

To launch experiments for only one dataset `i` (where `i` can be either `1`, `2`, `3` or `4` (`1`: Adult, `2`: COMPAS, `3`: Marketing, `4`: Default Credit))and one value of epsilon `eps` (minimum fairness acceptable):

```
srun python bench_mpi_new.py --dataset=i --epsilon=eps --debug=0
```

This script uses MPI to parallelize the computation of the three methods, for all 5 metrics and performing 5-folds cross validation.

## Merging results - Step 1

Raw results are located in the `results-test/` folder. A save of the obtained ones is available in the `results-heuristic-sample_robust_fairness` folder. Then :

- `source merge_all.sh` generates the compiled files for all metrics and all datasets.
- `source merge_all.sh i` generates the compiled files for dataset `i` only (where `i` can be either `1`, `2`, `3` or `4` (`1`: Adult, `2`: COMPAS, `3`: Marketing, `4`: Default Credit))

Compiled results are generated under `results_compiled`. They consist in seven files, for each metric, for each dataset. These files contain:

- the training Pareto Frontier for the three methods in three separate files
- the test Pareto Frontier for the three methods in three separate files
- training and test unfairness for the three methods in a single file

We left these files so that the Figures can be generated without re-running all the experiments.

## Merging results - Step 2

```
python merged.py
```

This script uses the summary files contained in `results_compiled` and merges them. It creates (in `results_merged`) three files per dataset per metric, containing data for all three methods :

- the training Pareto Frontier for the three methods 
- the test Pareto Frontier for the three methods 
- training and test unfairness for the three methods

## Plotting results (generating the Figures presented in the paper)

The R script we use needs the `ggplot2`, `scales`, `dplyr` and `ggpubr` R packages to be installed. It uses previously merged files (contained in `results_merged`) to generate the different Figures.

Running the R script creates all the Figures presented in the paper. To run the script:

```
Rscript results_plot.R
```

Generated plots are saved in the `graphs` folder. They consist, for each dataset and each metric, in (results for the three methods are presented jointly in each Figure):

- the training Pareto Frontier
- the test Pareto Frontier
- unfairness generalization plot: unfairness (test) = f(unfairness (train))

## Sample Robustness Audit

The audit of the built models can be done:

* using `build_compute_fairness_sample_robustness_heuristic.py`, which uses the Greedy algorithm to approximate sample-robustness.
The argument are `--metric` and `--dataset` and the audit is performed in a loop over a fixed range for epsilon.
The Greedy algorithm is implemented by `greedy_sample_robustness_auditor.py`.
The audit results are named as `dataset_DRO_compile_eps_metricID_BFS-objective-aware_sample_greedy_robustness_train_epsilon_solver_OR-tools.csv`.

* using `build_compute_fairness_sample_robustness.py`, which directly calls the sample robustness auditor of FairCORELS (requiring the modified FairCORELS including our exact version of sample-robustness to be installed).
The argument are `--metric` and `--dataset` and the audit is performed in a loop over a fixed range for epsilon.
The audit results are named as `dataset_DRO_compile_eps_metricID_BFS-objective-aware_sample_corrected_robustness_train_epsilon_solver_OR-tools.csv`.

Both save the audit results within `results_compiled`.

To plot the audit results (both exact and using the Greedy algorithm): `source gen_all_figures_sample_robustness.sh`
It calls the `plot_samplerobustness_with_greedy.py` for all datasets and metrics. The `--eliminate_trivials` argument only displays the audits until trivial models are built (then it stops at the previous value of epsilon).
The built figures are saved within the `Figures_SampleRobustness` folder.
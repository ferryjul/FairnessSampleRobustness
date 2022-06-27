The `faircorels` folder contains the modified version of the FairCORELS package (including the exact sample-robust method).
Install it with 
```
python setup.py install
```

The `faircorels/examples` folder contains the experiments files and results.
In particular:

* The `batch_all.sh` file can be used to get results for all 4 metrics, 4 values of epsilon, without validation set.
It calls the `sample_robust_frontier_epsilon_expes_script_mpi.py.py` script, which builds the entire sample robust frontier for one value of epsilon for one metric and one dataset.

* The `batch_all_validation.sh` file can be used to get results for all 4 metrics, 4 values of epsilon, with a validation set.
It calls the `sample_robustness_with_validation_mpi.py` Python script, which builds the sample robust frontier until all validations topping criteria are met, for one value of epsilon for one metric and one dataset.
 
* The `results_graham` folder contains all generated results.

* The `check_rerun_sample_robust.py` script can be used to check if all results are present (and eventually determine which ones are missing). Call it with `--val=1` to check results using a validation set, `--val=0` otherwise.

* The `wrapper_plot_all_paretos_test_validation_masks_frontier_val.sh` generates all Figures leveraging 5 parallel threads.
It calls the `plot_script_clean_new.py` Python script.
The arguments of this Python script are: `--metric`, `--epsilon`, `--dataset` that determine the associated parameters.
Parameter `--show=1` displays the figures, `--show=0` saves them.
Parameter `--plotType` determines the type of plot: 6 for the unfairness (train, test and validation when applicable) evolution through iterations, 7 for sample robustness (1D plot), 11 for test unfairness as a function of test error, and 42 for Train sample-robustness and train accuracy through iterations.
Doubling the script number, you generate the legends associated to these experiments (which can not be displayed directly - only saved).
Note that when also plotting results for the integration of our heuristic method within FairCORELS, such results are read in the appropriate folder (in `../../../FairCORELS_Heuristic/experiments/`)
* The `Paretos_All_Figures` folder contains all the generated Figures.
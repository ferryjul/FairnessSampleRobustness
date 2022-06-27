
To run the experiments, the TensorFlow Constrained Optimization library has to be installed, either from source code (https://github.com/google-research/tensorflow_constrained_optimization) or using Pypi (`pip install tensorflow-constrained-optimization`).

Additionally, Python module `six` (https://pypi.org/project/six/) is required.

# To launch experiments (using Slurm on a computing grid):

  

- For all expes: `./batch_all.sh`

Important: Before launching experiments for the Marketing and Default of Credit Card Clients dataset it is necessary to unzip the `data.zip` folder.

  

# To generate tables:

  

- For a single expe (both algorithms): `./merge_all.sh x` where `x` is in {1, 2, 3, 4} for one of the four expes

  

- For all expes (both algorithms): `./merge_all.sh`

  

- For one algorithm or the other ONLY: use above commands and modify variable "algos" in file `merge_results_eps0.py`

  

Calls to these scripts generate the corresponding .csv summary files (containing the results presented in the paper's tables) in the `summary` folder. Files corresponding to the results presented in the paper are left.

  

In the generated .csv summary files, these methods' results appear in the order of their modes (**unconstrained**, **baseline**, **validation**, **dromasks-10**, **dromasks-30**, **dromasks-50**), which is precisely the order used in the paper's Tables.

  

## Notes

  

The different methods (**unconstrained**, **baseline**, **validation**, **dromasks-10**, **dromasks-30**, **dromasks-50**) are run using the same scripts, providing a different value to the `mode` parameter:

  

* mode == 0: **unconstrained**

* mode == 1: **baseline**

* mode == 2: **validation**

* mode == 3: **dromasks-10**

* mode == 4: **dromasks-30**

* mode == 5: **dromasks-50**

  

Methods **unconstrained**, **baseline** and **validation** are implemented based on the source code and examples of https://github.com/google-research/tensorflow_constrained_optimization.

  

Note that when not using slurm, the different experiments can be launched individually using one of the four Python scripts and their parameters, eg:

  

```

python3 marketing_equalFPR-preproc-correct.py --mode=0 --algo=3 --drovalseed=100

```


launches the experiment on the Bank Marketing dataset, for the `unconstrained` method (mode 0), using algorithm 3 (Proxy Lagrangian) with random seed 100.

Results are saved in separate folders for the four experiments.

Algorithm 3 corresponds to the Proxy Lagrangian Approach, and Algorithm 4 to the usual Lagrangian one. Both are presented in the following paper:

  
```
Cotter, A., Jiang, H., & Sridharan, K. (2019, March). Two-player games for efficient non-convex constrained optimization. In Algorithmic Learning Theory (pp. 300-332). PMLR.
```
  

Finally, note that experiments presented in our paper are experiments 1 (on the Adult Income dataset), 2 (on the COMPAS dataset), 3 (on the Bank Marketing dataset) and 4 (on the Default of Credit Card Clients dataset). We compare our approach with:

  
```
Cotter, A., Gupta, M., Jiang, H., Srebro, N., Sridharan, K., Wang, S., ... & You, S. (2019, May). Training well-generalizing classifiers for fairness metrics and other data-dependent constraints. In International Conference on Machine Learning (pp. 1397-1405). PMLR.
```
  
## Detail of the files:

For experiments 1 and 2, the batch is performed separately for each algo/mode. 
Hence, for both the Adult and COMPAS datasets experiments, there are 12 batch files named with the convention: `dro_datasetexpe-modeID-algoID.sh`.
These individual batch files are called for algo 3 and 4 separately by scripts `batch_all_algo3.sh` and `batch_all_algo4.sh`.
These two scripts are called within `batch_all.sh`.

For experiments 3 and 4, a single batch file (resp. `batch_marketing_new_equalFPR.sh` and `batch_default_credit_min_tpr.sh`) is needed as parallel processing was improved. These batches are also called within `batch_all.sh`.

The experiments are done using one Python file for each experiment:

* experiment 1: `adult_ppr0.8.py`
* experiment 2: `compas_tpr0.05.py`
* experiment 3: `marketing_equalFPR-preproc-correct.py`
* experiment 4: `default_credit_max_FPR_and_min_TPR.py`

The results are saved within separate folders (one for each experiment).

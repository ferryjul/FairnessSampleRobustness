import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--val', type=int, default=0, help='0 for expes, 1 for expes with val')

args = parser.parse_args()

val = args.val

def check_triviality_reached_all_folds(sample_r_folds):
    for v in sample_r_folds:
        if v < 1.0:
            return False
    return True


def check_all_ok(tab1, tab2):
    for f in range(5):
        if not tab1[f]:
            return False
        if not tab2[f]:
            return False
    return True

epsilonsList = [0.985]#[0.98, 0.985, 0.99, 0.995]
metricsList = [1,3,4,5]
datasetsList = ["adult", "compas", "default_credit", "marketing"]
cart_prod = []
for e in epsilonsList:
    for m in metricsList:
        for d in datasetsList:
            cart_prod.append([e,m,d])

useLB = False
showEnded=True
expe = 0
if val == 0:
    expes_val = False
else:
    expes_val = True
if expes_val:
    folder='results_with_validation'
    val_prefix = '_with-validation'
else:
    folder='results'
    val_prefix = ''
expes_to_rerun = []
for epsilon, fairnessMetric, dataset in cart_prod:
    # First find last successful step
    frontier_id = 0
    found = True
    if expes_val:
        foldOK_Epsilon = [False, False, False, False, False]
        foldOK_Train = [False, False, False, False, False]
    while found: 
        fileName = './results_graham/%s/sample_robust_frontier%s_%s_%d_faircorels_eps%f_metric%d_LB%d.csv' %(folder, val_prefix, dataset, frontier_id, epsilon, fairnessMetric, useLB)
        try:
            fileContent = pd.read_csv('%s' %(fileName))
            if expes_val:
                for f in range(5):
                    if fileContent.values[f][12] <= 1.0 - epsilon:
                        foldOK_Epsilon[f] = True
                    if fileContent.values[f][12] <= fileContent.values[f][2]:
                        foldOK_Train[f] = True
            found = True
            frontier_id+=1
        except FileNotFoundError:
            found = False
    frontier_id-=1
    if frontier_id < 0:
        print("No file found for expe %d (dataset %s, metric %d, epsilon %f)." %(expe, dataset, fairnessMetric, epsilon))
        expes_to_rerun.append(expe)
    else:
        # Then retrieve values
        fileName = './results_graham/%s/sample_robust_frontier%s_%s_%d_faircorels_eps%f_metric%d_LB%d.csv' %(folder, val_prefix, dataset, frontier_id, epsilon, fairnessMetric, useLB)
        fileContent = pd.read_csv('%s' %(fileName))
        sample_r_values=[fileContent.values[l][4] for l in range(5)]
        if expes_val:
            val_unf=fileContent.values[5][12]
            unfairnessT=fileContent.values[5][2]
            condition = not (check_triviality_reached_all_folds(sample_r_values) or check_all_ok(foldOK_Epsilon, foldOK_Train))
        else:
            condition = not check_triviality_reached_all_folds(sample_r_values)
        if condition:
            expes_to_rerun.append(expe)
            print("Found %d files for expe %d (dataset %s, metric %d, epsilon %f)." %(frontier_id, expe, dataset, fairnessMetric, epsilon))
            print(sample_r_values)
        else:
            if showEnded:
                print("Expe is complete! Found %d files for expe %d (dataset %s, metric %d, epsilon %f)." %(frontier_id, expe, dataset, fairnessMetric, epsilon))
    expe+=1
print("Need to rerun %d expes:" %len(expes_to_rerun))
#print(expes_to_rerun)
print('[', end='')
for i,e in enumerate(expes_to_rerun):
    if i == len(expes_to_rerun)-1:
        print(e, end=']\n')    
    else:
        print(e, end=',')

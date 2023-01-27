import numpy as np
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import argparse
from faircorels import load_from_csv, FairCorelsClassifier, ConfusionMatrix, Metric, SampleRobustnessAuditor
import csv
import time
import pandas as pd
from mpi4py import MPI
import sys

N_ITER = 25*10**5 # The maximum number of nodes in the prefix tree
sensitive_attr_column = 0 # Column of the dataset used to define protected group membership (=> sensitive attribute)
unsensitive_attr_column = 1 
prediction_name_dict = {
    "compas" : "(recidivism:yes)",
    "adult" : "(income > 50K)",
    "marketing" : "(subscribe:yes)",
    "german_credit": "(credit_rating:yes)",
    "default_credit": "(default_payment:yes)"
}

# parser initialization
parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--expe', type=int, default=0, help='expe id (cart product index)')

epsilonsList = [0.98, 0.985, 0.99, 0.995]
metricsList = [1,3,4,5]
datasetsList = ["adult", "compas", "default_credit", "marketing"]
cart_prod = []
for e in epsilonsList:
    for m in metricsList:
        for d in datasetsList:
            cart_prod.append([e,m,d])

args = parser.parse_args()
epsilon = cart_prod[args.expe][0]
fairnessMetric =  cart_prod[args.expe][1]
dataset =  cart_prod[args.expe][2]
print("Expe %d/%d: dataset %s, metric %d, epsilon %f" %(args.expe, len(cart_prod), dataset, fairnessMetric, epsilon))
sys.stdout.flush()
#exit()
n_workers = 8
useLB = False
lambdaParam = 1e-3 # The regularization parameter penalizing rule lists length
modeArg = 6
initial_min_sample_robustness_arg=0.0

# X, y, features, prediction = load_from_csv("./data/%s_rules_full.csv" %dataset)#("./data/adult_full.csv") # Load the dataset
X, y, features, prediction = load_from_csv("./data/%s_fullRules_gen.csv" %dataset)#("./data/adult_full.csv") # Load the dataset

#print("Sensitive attribute is ", features[sensitive_attr_column])
#print("Unsensitive attribute is ", features[unsensitive_attr_column])


# We prepare the folds for our 5-folds cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    folds.append([X_train, y_train, X_test, y_test])

accuracy = []
unfairness = []

def compute_unfairness(sensVect, unSensVect, y, y_pred):
    cm = ConfusionMatrix(sensVect, unSensVect, y_pred, y)
    cm_minority, cm_majority = cm.get_matrix()
    fm = Metric(cm_minority, cm_majority)

    if fairnessMetric == 1:
        unf = fm.statistical_parity()
    elif fairnessMetric == 2:
        unf = fm.predictive_parity()
    elif fairnessMetric == 3:
        unf = fm.predictive_equality()
    elif fairnessMetric == 4:
        unf = fm.equal_opportunity()
    elif fairnessMetric == 5:
        unf = fm.equalized_odds()
    elif fairnessMetric == 6:
        unf = fm.conditional_use_accuracy_equality()
    else:
        unf = -1
    
    return unf

# returns s_a, s_b, z_a, z_b for the given metric
def compute_values(sensVect, unSensVect, y, y_pred, theMetric):
    cm = ConfusionMatrix(sensVect, unSensVect, y_pred, y)
    cm_minority, cm_majority = cm.get_matrix()
    fm = Metric(cm_minority, cm_majority)

    if theMetric == 1:
        return int(cm_minority['TP']+cm_minority['FP']), int(cm_majority['TP']+cm_majority['FP']), int(cm_minority['TP']+cm_minority['FP']+cm_minority['FN']+cm_minority['TN']), int(cm_majority['TP']+cm_majority['FP']+cm_majority['FN']+cm_majority['TN'])
    elif theMetric == 2:
        print("Should not happen")
        exit()
    elif theMetric == 3:
        return int(cm_minority['FP']), int(cm_majority['FP']), int(cm_minority['FP']+cm_minority['TN']), int(cm_majority['FP']+cm_majority['TN'])
    elif theMetric == 4:
        return int(cm_minority['FN']), int(cm_majority['FN']), int(cm_minority['TP']+cm_minority['FN']), int(cm_majority['TP']+cm_majority['FN'])
    elif theMetric == 5:
        print("need to do calls differently for EOdds")
    elif theMetric == 6:
        print("Should not happen")
        exit()
    else:
        unf = -1
    
    return unf

def oneFold(foldIndex, X_fold_data): # This part could be multithreaded for better performance
    X_train, y_train, X_test, y_test = X_fold_data

    # Separate protected features to avoid disparate treatment
    # - Training set
    sensVect_train =  X_train[:,sensitive_attr_column]
    unSensVect_train =  X_train[:,unsensitive_attr_column] 
    X_train_unprotected = X_train[:,2:]

    # - Test set
    sensVect_test =  X_test[:,sensitive_attr_column]
    unSensVect_test =  X_test[:,unsensitive_attr_column] 
    X_test_unprotected = X_test[:,2:]

    # Create the FairCorelsClassifier object
    clf = FairCorelsClassifier(n_iter=N_ITER,
                            c=lambdaParam, # sparsity regularization parameter
                            max_card=1, # one rule = one attribute
                            policy="bfs", # exploration heuristic: BFS
                            bfs_mode=2, # type of BFS: objective-aware
                            mode=modeArg, # epsilon-constrained mode
                            useUnfairnessLB=useLB,
                            forbidSensAttr=False,
                            fairness=fairnessMetric, 
                            epsilon=epsilon, # fairness constrait
                            verbosity=[], # don't print anything
                            maj_vect=unSensVect_train, # vector defining unprotected group
                            min_vect=sensVect_train, # vector defining protected group
                            min_support = 0.01,
                            min_sample_robustness=current_sample_robustness[foldIndex],
                            sample_robustness_auditor_object=auditor
                            )
    # Train it
    clf.fit(X_train_unprotected, y_train, features=features[2:], prediction_name=prediction_name_dict[dataset])#, time_limit = 10)# max_evals=100000) # time_limit=8100, 

    # Print the fitted model
    print("Fold ", foldIndex, " :", clf.rl_)

    # Evaluate our model's accuracy
    accTraining = clf.score(X_train_unprotected, y_train)
    accTest = clf.score(X_test_unprotected, y_test)

    # Evaluate our model's fairness
    train_preds = clf.predict(X_train_unprotected)
    unfTraining = compute_unfairness(sensVect_train, unSensVect_train, y_train, train_preds)

    test_preds = clf.predict(X_test_unprotected)
    unfTest = compute_unfairness(sensVect_test, unSensVect_test, y_test, test_preds)

    # Also compute/collect additional parameters
    length = len(clf.rl_.rules)-1 # -1 because we do not count the default rule
    objF = ((1-accTraining) + (lambdaParam*length)) # best objective function reached
    exploredBeforeBest = int(clf.nbExplored)
    cacheSizeAtExit = int(clf.nbCache)
    if fairnessMetric in [1,3,4]:
        s_a, s_b, z_a, z_b = compute_values(sensVect_train, unSensVect_train, y_train, train_preds, fairnessMetric)
        fairness_sample_robustness = auditor.audit_robustness(s_a, s_b, z_a, z_b, 1.0-epsilon, debug=debug_arg, solver=solver_arg)
    elif fairnessMetric == 5:
        s_a, s_b, z_a, z_b = compute_values(sensVect_train, unSensVect_train, y_train, train_preds, 3)
        fairness_sample_robustness_pe = auditor.audit_robustness(s_a, s_b, z_a, z_b, 1.0-epsilon, debug=debug_arg, solver=solver_arg)
        s_a, s_b, z_a, z_b = compute_values(sensVect_train, unSensVect_train, y_train, train_preds, 4)
        fairness_sample_robustness_eo = auditor.audit_robustness(s_a, s_b, z_a, z_b, 1.0-epsilon, debug=debug_arg, solver=solver_arg)
        fairness_sample_robustness_vals = [fairness_sample_robustness_pe, fairness_sample_robustness_eo]
        fairness_sample_robustness_vals_distances = [fairness_sample_robustness_pe.jaccard_dist, fairness_sample_robustness_eo.jaccard_dist]
        fairness_sample_robustness=fairness_sample_robustness_vals[fairness_sample_robustness_vals_distances.index(min(fairness_sample_robustness_vals_distances))]
    else:
        print("Sample robustness audit not implemented for metric ", fairnessMetric, "!")
    print("Training set unfairness = ", unfTraining, ", test set unfairness = ", unfTest, ", relative gap = ", (unfTest - unfTraining)/unfTraining)
    print("Training set accuracy = ",accTraining, ", test set accuracy = ", accTest, ", relative gap = ", (accTest - accTraining)/accTraining)
    print("[Fold %d] Sample robustness is " %foldIndex, fairness_sample_robustness.jaccard_dist)
    print("Detailed result: x_a = ", fairness_sample_robustness.x_a, ", y_a = ", fairness_sample_robustness.y_a, ", x_b = ", fairness_sample_robustness.x_b, ", y_b = ", fairness_sample_robustness.y_b, ",score = ", fairness_sample_robustness.score, ", status = ", fairness_sample_robustness.status, ", unfairness = ", fairness_sample_robustness.unfairness, ", jaccard_dist = ", fairness_sample_robustness.jaccard_dist)
    # exit()
    sys.stdout.flush()
    return [foldIndex, accTraining, unfTraining, objF, accTest, unfTest, exploredBeforeBest, cacheSizeAtExit, length, fairness_sample_robustness.jaccard_dist, clf.rl_]

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
start = time.clock()
auditor = SampleRobustnessAuditor(1)
solvers = ['Mistral', 'OR-tools']
solver_arg = solvers[1]
debug_arg = 0
frontier_id = 0

current_sample_robustness = []
for i in range(5):
    current_sample_robustness.append(initial_min_sample_robustness_arg)

def check_triviality_reached_all_folds(sample_r_folds):
    for v in sample_r_folds:
        if v < 1.0:
            return False
    return True
mustStop = False
# try to retrieve past results:
found = True
while found: 
    fileName = './results/sample_robust_frontier_%s_%d_faircorels_eps%f_metric%d_LB%d.csv' %(dataset, frontier_id, epsilon, fairnessMetric, useLB)
    try:
        fileContent = pd.read_csv('%s' %(fileName))
        found = True
        frontier_id+=1
    except FileNotFoundError:
        found = False
frontier_id-=1
if frontier_id >= 0:
    # Then retrieve values
    fileName = './results/sample_robust_frontier_%s_%d_faircorels_eps%f_metric%d_LB%d.csv' %(dataset, frontier_id, epsilon, fairnessMetric, useLB)
    fileContent = pd.read_csv('%s' %(fileName))
    if rank == 0:
        print("Found %d files for this expe (dataset %s, metric %d, epsilon %f)." %(frontier_id, dataset, fairnessMetric, epsilon))
        print("Retrieved results with sample robutness : ", current_sample_robustness)
        sys.stdout.flush()
    current_sample_robustness=[fileContent.values[l][4] for l in range(5)]
frontier_id+=1

while not mustStop:
    if rank == 0:
        print("Step %d, current_sample_robustness =  " %frontier_id, current_sample_robustness)
        sys.stdout.flush()
    # Run training/evaluation for all folds using multi-threading
    #ret = Parallel(n_jobs=n_workers)(delayed(oneFold)(foldIndex, X_fold_data) for foldIndex, X_fold_data in enumerate(folds))
    result = oneFold(rank, folds[rank])
    end = time.clock() 
    current_sample_robustness[rank]=result[9]
    print("--> Total training time is ", end - start, " seconds.")

    ret = COMM.gather(result, root=0)

    if rank == 0:
        # Unwrap the results
        accuracy = [ret[i][4] for i in range(0,5)]
        unfairness = [ret[i][5] for i in range(0,5)]
        objective_functions = [ret[i][3] for i in range(0,5)]
        accuracyT = [ret[i][1] for i in range(0,5)]
        unfairnessT = [ret[i][2] for i in range(0,5)]
        #Save results in a csv file
        resPerFold = dict()
        for aRes in ret:
            resPerFold[aRes[0]] = [aRes[1], aRes[2], aRes[3], aRes[4], aRes[5], aRes[6], aRes[7], aRes[8], aRes[9], aRes[10]]
            current_sample_robustness[aRes[0]] = aRes[9]
        print("------------------- Frontier point %d -----------------------------------------" %frontier_id)
        print("=========> Training Accuracy (average)= ", np.average(accuracyT))
        print("=========> Training Unfairness (average)= ", np.average(unfairnessT))
        print("=========> Training Objective function value (average)= ", np.average(objective_functions))
        print("=========> Test Accuracy (average)= ", np.average(accuracy))
        print("=========> Test Unfairness (average)= ", np.average(unfairness))
        print("=========> Training Sample Robustness (average)= ", np.average([resPerFold[index][8] for index in range(5)]))
   
        with open('./results/sample_robust_frontier_%s_%d_faircorels_eps%f_metric%d_LB%d.csv' %(dataset, frontier_id, epsilon, fairnessMetric, useLB), mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Fold#', 'Training accuracy', 'Training Unfairness(%d)' %fairnessMetric, 'Training objective function', 'Training Sample-Robustness', 'Test accuracy', 'Test unfairness', '#Nodes explored for best solution', 'Cache size for best solution', 'Average length', 'RL'])#, 'Fairness STD', 'Accuracy STD'])
            for index in range(5):
                csv_writer.writerow([index, resPerFold[index][0], resPerFold[index][1], resPerFold[index][2], resPerFold[index][8], resPerFold[index][3], resPerFold[index][4], resPerFold[index][5], resPerFold[index][6], resPerFold[index][7], resPerFold[index][9]])#,fairnessSTD_list_test_tot[i][fairnessMetric-1], accuracySTD_list_test_tot[i]])
            csv_writer.writerow(['Average', np.average(accuracyT), np.average(unfairnessT), np.average(objective_functions), np.average([resPerFold[index][8] for index in range(5)]), np.average(accuracy), np.average(unfairness), np.average([resPerFold[index][6] for index in range(5)]), np.average([resPerFold[index][7] for index in range(5)]), ''])#,fairnessSTD_list_test_tot[i][fairnessMetric-1], accuracySTD_list_test_tot[i]])
    
        if check_triviality_reached_all_folds(current_sample_robustness):
            mustStop = True
        else:
            mustStop = False

    mustStop = COMM.scatter([mustStop,mustStop,mustStop,mustStop,mustStop], root=0)
    frontier_id += 1

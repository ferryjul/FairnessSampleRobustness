import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from joblib import Parallel, delayed, parallel_backend
from faircorels import load_from_csv, CorelsClassifier, ConfusionMatrix, Metric
import csv
import argparse
import os
from config import get_data, get_metric, get_strategy
from mpi4py import MPI


# parser initialization
parser = argparse.ArgumentParser(description='Evaluation of FairCORELS')
parser.add_argument('--dataset', type=int, default=1, help='Dataset. 1: Adult, 2: COMPAS, 3: Marketing, 4: Default Credit')
parser.add_argument('--epsilon', type=int, default=0, help='Epsilon value (index in array)')
parser.add_argument('--debug', type=int, default=0, help='Print additional information. 1: Yes, 0: No')

#parser.add_argument('--strat', type=int, default=1, help='Search strategy. 1: bfs, 2:curious, 3: lower_bound, 4: bfs_objective_aware')
args = parser.parse_args()

#get dataset and relative infos
dataset, decision, prediction_name, min_feature, min_pos, maj_feature, maj_pos = get_data(args.dataset)

#------------------------setup config

#iterations
N_ITER = 25*10**5
n_masks_list = [0,10,30]#, 30]#,30]#args.nbMasks
lambdaVal = 1e-3
debugMode = args.debug
#fairness constraint
metricsList = [5,6]#[1, 2, 3, 4, 5]
#epsilons
#epsilons
epsilon_range1 = np.arange(0.705, 0.901, 0.005)
epsilon_range2 = np.arange(0.902, 0.98, 0.002)
epsilon_range3 = np.arange(0.98, 0.9895, 0.001)
epsilon_range4 = np.arange(0.99, 1.000, 0.0002)
epsilon_range = list(epsilon_range1) + list(epsilon_range2) + list(epsilon_range3) + list(epsilon_range4)
base = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]
epsilon_range = base + epsilon_range
epsilons = [round(x,4) for x in epsilon_range] #147 values
epsilonVal = epsilons[args.epsilon] # 0 -> 146
if debugMode:
    epsilons = [0.0, 0.25, 0.5, 0.75, 1.0] #327 values
    epsilonVal = epsilons[args.epsilon]
    print("Using %d values for epsilon." %len(epsilons))
# use filtering
filtering = False
suffix = "without_filtering"

# get search strategy
#strategy, bfsMode, strategy_name = get_strategy(args.strat)
strategy = "bfs"
bfsMode = 2
strategy_name = "BFS-objective-aware"

#save direcory
#save_dir = "./results/{}_{}/{}".format(dataset, suffix, strategy_name)
#os.makedirs(save_dir, exist_ok=True)


# load dataset
X, y, features, prediction = load_from_csv("./data/{}_fullRules_gen.csv".format(dataset,dataset))


# creating k-folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)


folds = []
i=0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    folds.append([X_train, y_train, X_test, y_test, i])
    i +=1

cart_product = []
for m in n_masks_list:
    for fold in folds:
        for f_metric in metricsList:
            cart_product.append([m, fold, f_metric])

#print("------------------------------------------------------->>>>>>>> {}".format(len(cart_product)))
if debugMode==1:
    print("Sensitive attribute is ", features[min_pos], "(expected - %s)" %min_feature)
    print("Unsensitive attribute is ", features[maj_pos], "(expected - %s)" %maj_feature)

def compute_unfairness(fm, fairnessMetric):
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


def fit(fold, epsilon, fairness, n_masks):
    X_train, y_train, X_test, y_test, fold_id = fold[0], fold[1], fold[2], fold[3], fold[4]
    # Prepare the vectors defining the protected (sensitive) and unprotected (unsensitive) groups
    # Uncomment the print to get information about the sensitive/unsensitive vectors
    sensVect =  X_train[:,min_pos] # 32 = female 
    unSensVect =  X_train[:,maj_pos] # 33 = male (note that here we could directly give the column number)

    if debugMode==1:
        unique, counts = np.unique(sensVect, return_counts=True)
        sensDict = dict(zip(unique, counts))
        print("Sensitive vector captures %d instances" %sensDict[1])
        unique2, counts2 = np.unique(unSensVect, return_counts=True)
        unSensDict = dict(zip(unique2, counts2))
        print("Unsensitive vector captures %d instances" %unSensDict[1])

    clf = CorelsClassifier(n_iter=N_ITER, 
                            min_support=0.01,
                            c=lambdaVal, 
                            max_card=1, 
                            policy=strategy,
                            bfs_mode=bfsMode,
                            mode=3,
                            useUnfairnessLB=filtering,
                            forbidSensAttr=False,
                            fairness=fairness, 
                            epsilon=epsilon,
                            #maj_pos=maj_pos, 
                            #min_pos=min_pos,
                            maj_vect=unSensVect, 
                            min_vect=sensVect,
                            verbosity=[]
                            )

    clf.fit(X_train[:,2:], y_train, features=features[2:], prediction_name=prediction_name, nb_masks = n_masks)

    
    #test
    df_test = pd.DataFrame(X_test, columns=features)
    df_test[decision] = y_test
    df_test["predictions"] = clf.predict(X_test[:,2:])
    cm_test = ConfusionMatrix(df_test[min_feature], df_test[maj_feature], df_test["predictions"], df_test[decision])
    cm_minority_test, cm_majority_test = cm_test.get_matrix()
    fm_test = Metric(cm_minority_test, cm_majority_test)

    #train 
    df_train = pd.DataFrame(X_train, columns=features)
    df_train[decision] = y_train
    df_train["predictions"] = clf.predict(X_train[:,2:])
    cm_train = ConfusionMatrix(df_train[min_feature], df_train[maj_feature], df_train["predictions"], df_train[decision])
    cm_minority_train, cm_majority_train = cm_train.get_matrix()
    fm_train = Metric(cm_minority_train, cm_majority_train)

    acc_test = clf.score(X_test[:,2:], y_test)
    unf_test = compute_unfairness(fm_test, fairness) #fm_test.fairness_metric(fairness)
    acc_train = clf.score(X_train[:,2:], y_train)
    unf_train = compute_unfairness(fm_train, fairness) #fm_train.fairness_metric(fairness)
    # Retrieve objective function value
    length = len(clf.rl_.rules)-1
    objF = ((1-acc_train) + (lambdaVal*length))

    mdl = clf.rl().__str__()


    return [fold_id, fairness, acc_train, unf_train, objF, acc_test, unf_test, int(clf.nbExplored), int(clf.nbCache), length, mdl, n_masks]

def split(container, count):
    return [container[_i::count] for _i in range(count)]

def process_results(fairness_list, results, epsilon):
    # save all fold - eps in dict
    res_dict = {}
    row_list = []
    row_list_model = []

    for res in results:
        res_dict[str(res[0]) + "_" + str(res[1]) + "_" + str(res[11])] = res

    for nb_masks in n_masks_list:
        for fairness in fairness_list:
            accuracy_test = []
            unfairness_test = []
            accuracy_train = []
            unfairness_train = []
            objFList = []
            nbNodesList = []
            cacheSizeList = []
            lenList = []
            mdlList = []
            row = {}
            row_model = {}

            for fold_id in [0, 1, 2, 3, 4]:
                key = str(fold_id) + "_" + str(fairness) + "_" + str(nb_masks)
                result = res_dict[key]
                acc_train, unf_train, objF, acc_test, unf_test, nbExplored, nbCache, length, mdl = result[2], result[3], result[4], result[5], result[6],result[7], result[8], result[9], result[10]
                
                accuracy_test.append(acc_test)
                accuracy_train.append(acc_train)

                unfairness_test.append(unf_test)
                unfairness_train.append(unf_train)
                objFList.append(objF)
                nbNodesList.append(nbExplored)
                cacheSizeList.append(nbCache)
                lenList.append(length)
                mdlList.append(mdl)

            with open('./results-test/%s_eps%f_metric%d_%s_%s_%dmasks.csv' %(dataset, epsilon, fairness, strategy_name, suffix, nb_masks), mode='w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['Fold#', 'Training accuracy', 'Training Unfairness', 'Training objective function', 'Test accuracy', 'Test unfairness', '#Nodes explored for best solution', 'Cache size for best solution', 'RL length', 'RL'])#, 'Fairness STD', 'Accuracy STD'])
                for index in range(5):
                    csv_writer.writerow([index, accuracy_train[index], unfairness_train[index], objFList[index], accuracy_test[index], unfairness_test[index], nbNodesList[index], cacheSizeList[index], lenList[index], mdlList[index]])#,fairnessSTD_list_test_tot[i][fairnessMetric-1], accuracySTD_list_test_tot[i]])
                csv_writer.writerow(['Average', np.mean(accuracy_train), np.mean(unfairness_train), np.mean(objFList), np.mean(accuracy_test),  np.mean(unfairness_test), '', '','', ''])#,fairnessSTD_list_test_tot[i][fairnessMetric-1], accuracySTD_list_test_tot[i]])

'''results = []
for params in cart_product:
    masks = params[0]
    fold = params[1]
    epsilon = params[2]
    results.append(fit(fold, epsilon, fairness_metric, masks))

process_results(epsilons, results, fairness_metric)
'''

COMM = MPI.COMM_WORLD

if COMM.rank == 0:
    jobs = split(cart_product, COMM.size)
else:
    jobs = None

jobs = COMM.scatter(jobs, root=0)



results = []
for job in jobs:
    masks = job[0]
    fold = job[1]
    fairness_metric = job[2]
    #print("----"*20 + ">>> fold: {}, epsilon: {}".format(fold[4], epsilon))
    results.append(fit(fold, epsilonVal, fairness_metric, masks))


# Gather results on rank 0.
results = MPI.COMM_WORLD.gather(results, root=0)

if COMM.rank == 0:
    results = [_i for temp in results for _i in temp]
    process_results(metricsList, results, epsilonVal)

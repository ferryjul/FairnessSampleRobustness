import pandas as pd 
import csv
import numpy as np
import argparse
from config import get_data, get_metric, get_strategy
from faircorels import load_from_csv
from sklearn.model_selection import KFold
from parse import *
import math
from faircorels import SampleRobustnessAuditor

parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 : statistical parity, 2 : predictive parity, 3 : predictive equality, 4 : equal opportunity, 5 : equalized odds, 6 : conditional use accuracy equality')
parser.add_argument('--dataset', type=int, default=1, help='Dataset. 1: Adult, 2: COMPAS, 3: Marketing, 4: Default Credit')
parser.add_argument('--filtering', type=int, default=0, help='Use improved CP filtering. 1: Yes, 0: No')

args = parser.parse_args()
dataset, decision, prediction_name, min_feature, min_pos, maj_feature, maj_pos = get_data(args.dataset)
suffix = "with_filtering" if args.filtering==1 else "without_filtering"

epsilon_range1 = []#np.arange(0.705, 0.901, 0.005)
epsilon_range2 = np.arange(0.902, 0.98, 0.002)
epsilon_range3 = np.arange(0.98, 0.9895, 0.001)
epsilon_range4 = np.arange(0.99, 1.000, 0.0002)
epsilon_range = list(epsilon_range1) + list(epsilon_range2) + list(epsilon_range3) + list(epsilon_range4)
base = []#[0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]
epsilon_range = base + epsilon_range
epsilons = [round(x,4) for x in epsilon_range] #147 values
#epsilons=[0.98, 0.985, 0.99, 0.995] # only values used for exact sample robustness experiments
fairnessMetric = args.metric#1

solvers = ['Mistral', 'OR-tools']
solver_arg = solvers[1]

print("Loading and preparing data...")
X, y, features, prediction = load_from_csv("./data/{}_fullRules_gen.csv".format(dataset,dataset))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = []
i=0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    folds.append([X_train, y_train, X_test, y_test, i])
    i +=1
print("...done")

auditor = SampleRobustnessAuditor()

def check_measure_numerator(label, pred, fMetric):
    if fMetric == 1:
        return pred
    elif fMetric == 3:
        if label == 0 and pred == 1: # FP
            return 1
        else:
            return 0
    elif fMetric == 4:
        if label == 1 and pred == 0: # FN
            return 1
        else:
            return 0
    else:
        print("unsupported metric. Exiting")
        exit()

def check_measure_denumerator(label, fMetric):
    if fMetric == 1:
        return 1
    elif fMetric == 3:
        if label == 0:
            return 1
        else:
            return 0
    elif fMetric == 4:
        return label
    else:
        print("unsupported metric. Exiting")
        exit()

def predict_from_rl(example, rl_str, features):
    rules = rl_str.split('\n')
    for arule in rules:
        if arule == "RULELIST:": # common header
            continue
        else:
            if "else if" in arule:
                format_string = '[{}]:'
                #print("arule=", arule)
                antecedent = search(format_string, arule).fixed[0]
                if antecedent is None:
                    print("error reading this rule -> ", arule)
                    exit()
                format_string = '[%s] = {}' %decision
                consequent = search(format_string, arule).fixed[0]
                #print("rule :", antecedent, "->", consequent)
                #print(antecedent)
                if example[features.index(antecedent)] == 1:
                    if consequent == 'T':
                        return 1
                    else:
                        return 0
            elif "if" in arule:
                format_string = '[{}]:'
                #print("arule=", arule)
                antecedent = search(format_string, arule).fixed[0]
                if antecedent is None:
                    print("error in antecedent reading this rule -> ", arule)
                    exit()
                format_string = '[%s] = {}' %decision
                consequent = search(format_string, arule).fixed[0]
                #print("rule :", antecedent, "->", consequent)
                #print(antecedent)
                if example[features.index(antecedent)] == 1:
                    if consequent == 'T':
                        return 1
                    elif consequent == 'F':
                        return 0
                    else:
                        print("error in consequent reading this rule -> ", arule)
                        exit()
            elif "else" in arule:
                format_string = '[%s] = {}' %decision
                consequent = search(format_string, arule).fixed[0]
                #print("default rule :", "->", consequent)
                if consequent == 'T':
                    return 1
                elif consequent == 'F':
                    return 0
                else:
                    print("error in consequent reading this rule -> ", arule)
                    exit()
            elif "=" in arule: # constant rule list (constant classifier)
                format_string = '[%s] = {}' %decision
                consequent = search(format_string, arule).fixed[0]
                #print("default rule :", "->", consequent)
                if consequent == 'T':
                    return 1
                elif consequent == 'F':
                    return 0
                else:
                    print("error in consequent reading this rule -> ", arule)
                    exit()
            else:
                print("error reading this rule -> ", arule)
                exit()
    return 0

def rl_string_robustness(rl_str, eps, fold_id, expected_fold_accuracy, expected_fold_unfairness, theMetric):
    fold = folds[fold_id]
    X_train, y_train, X_test, y_test, fold_id = fold[0], fold[1], fold[2], fold[3], fold[4]
    # Prepare the vectors defining the protected (sensitive) and unprotected (unsensitive) groups
    # Uncomment the print to get information about the sensitive/unsensitive vectors
    #sensVect =  X_train[:,min_pos] # 32 = female 
    #unSensVect =  X_train[:,maj_pos] # 33 = male (note that here we could directly give the column number)
    Z_a = 0 # total (sens)
    Z_b = 0 # total (unsens)
    S_a = 0 # numerator (sens)
    S_b = 0 # numerator (unsens)
    ok=0
    tot=0
    example_index = 0
    for example in X_train:
        label = y_train[example_index]
        pred = predict_from_rl(example, rl_str, features)
        if example[min_pos] == 1:
            Z_a+=check_measure_denumerator(label, theMetric)
            S_a += check_measure_numerator(label, pred, theMetric)
        elif example[maj_pos] == 1:
            Z_b +=check_measure_denumerator(label, theMetric)
            S_b += check_measure_numerator(label, pred, theMetric)
        else:
            print("error - example = ", example)
            exit()
        if pred == label:
            ok += 1
        example_index += 1
    unfairness_recomputed = abs((S_a/Z_a)-(S_b/Z_b))
    accuracy_recomputed = abs(ok/example_index)
    #print("read fold fairness=%f" %(unfairness_recomputed))
    #print("read fold accuracy=%f" %(accuracy_recomputed))
    measOK = True
    if not math.isclose(accuracy_recomputed, expected_fold_accuracy):
        print("error, expected accuracy ", expected_fold_accuracy, ", got accuracy = ", accuracy_recomputed)
    if not math.isclose(unfairness_recomputed, expected_fold_unfairness):
        measOK = False
        #print("error, expected unfairness ", expected_fold_unfairness, ", got unfairness = ", unfairness_recomputed)
        #print("detailed computation: ", "abs((%d/%d)-(%d/%d))" %(S_a,Z_a,S_b,Z_b))
    robustness_results = auditor.audit_robustness(int(S_a), int(S_b), int(Z_a), int(Z_b), (1.0-eps), debug=0, solver=solver_arg)
    rob = robustness_results.jaccard_dist
    del robustness_results
    return rob, measOK

trainSampleRobustness = [[], [], []]
trivialModelsCount = [[], [], []]
accTrains = [[], [], []]
accTests = [[], [], []]
unfTrains = [[], [], []]
unfTests = [[], [], []]

nb_masks1 = 0
nb_masks2 = 10
nb_masks3 = 30
heuristic = "BFS-objective-aware"
effectiveEpsilons = []
index = 0
tobererun = []
debug = 1
maxCache = 0
n_iter = 25*10**5
epsilons = np.asarray(epsilons)
epsilons=list(epsilons[epsilons < 1.0])#[0.999, 0.9992]#[0.9996]
#epsilons=[0.9998]
for epsilon in epsilons:
    #epsilon = 1.0 - epsilonV
    #try:
    ok1 = False
    ok2 = False
    ok3 = False
    data1 = pd.read_csv('./results-heuristic-sample_robust_fairness/%s_eps%f_metric%d_%s_%s_%dmasks.csv' %(dataset, epsilon, fairnessMetric, heuristic, suffix, nb_masks1))
    ok1 = True
    data2 = pd.read_csv('./results-heuristic-sample_robust_fairness/%s_eps%f_metric%d_%s_%s_%dmasks.csv' %(dataset, epsilon, fairnessMetric, heuristic, suffix, nb_masks2))
    ok2 = True
    data3 = pd.read_csv('./results-heuristic-sample_robust_fairness/%s_eps%f_metric%d_%s_%s_%dmasks.csv' %(dataset, epsilon, fairnessMetric, heuristic, suffix, nb_masks3))
    ok3 = True
    accTrains[0].append(data1.values[5][1])
    unfTrains[0].append(data1.values[5][2])
    accTests[0].append(data1.values[5][4])
    unfTests[0].append(data1.values[5][5])

    accTrains[1].append(data2.values[5][1])
    unfTrains[1].append(data2.values[5][2])
    accTests[1].append(data2.values[5][4])
    unfTests[1].append(data2.values[5][5])

    accTrains[2].append(data3.values[5][1])
    unfTrains[2].append(data3.values[5][2])
    accTests[2].append(data3.values[5][4])
    unfTests[2].append(data3.values[5][5])

    effectiveEpsilons.append(epsilon)
    #print("overall model unfairness 5-folds average=", data1.values[5][2])
    #print("overall model accuracy 5-folds average=", data1.values[5][1])
    if debug>0:
        rl1_robustness = []
        rl2_robustness = []
        rl3_robustness = []
        rl_trivial_nb_cnt = [0, 0, 0]
        for i in range(5):
            if data1.values[i][7] > maxCache:
                maxCache = data1.values[i][7]
            if data2.values[i][7] > maxCache:
                maxCache = data2.values[i][7]
            if data3.values[i][7] > maxCache:
                maxCache = data3.values[i][7]
            if len(data1.values[i][9].split('\n')) == 2:
                rl_trivial_nb_cnt[0]+=1
            if len(data2.values[i][9].split('\n')) == 2:
                rl_trivial_nb_cnt[1]+=1
            if len(data3.values[i][9].split('\n')) == 2:
                rl_trivial_nb_cnt[2]+=1
            if fairnessMetric in [1,3,4]:
                rl1_rob_res, rl1_rob_err = rl_string_robustness(data1.values[i][9], epsilon, i, data1.values[i][1], data1.values[i][2], fairnessMetric)
                if not rl1_rob_err:
                    print("Error in fairness computation vs expectation. Exiting.")
                    exit()
                rl1_robustness.append(rl1_rob_res)

                rl2_rob_res, rl2_rob_err = rl_string_robustness(data2.values[i][9], epsilon, i, data2.values[i][1], data2.values[i][2], fairnessMetric)
                rl2_robustness.append(rl2_rob_res)
                if not rl2_rob_err:
                    print("Error in fairness computation vs expectation. Exiting.")
                    exit()

                rl3_rob_res, rl3_rob_err = rl_string_robustness(data3.values[i][9], epsilon, i, data3.values[i][1], data3.values[i][2], fairnessMetric)
                rl3_robustness.append(rl3_rob_res)
                if not rl3_rob_err:
                    print("Error in fairness computation vs expectation. Exiting.")
                    exit()

            elif fairnessMetric == 5:
                rl1_pe_res, rl1_pe_status = rl_string_robustness(data1.values[i][9], epsilon, i, data1.values[i][1], data1.values[i][2], 3)
                rl1_eo_res, rl1_eo_status = rl_string_robustness(data1.values[i][9], epsilon, i, data1.values[i][1], data1.values[i][2], 4)
                rl1_eodds_comp = min(rl1_pe_res, rl1_eo_res)
                rl1_robustness.append(rl1_eodds_comp)
                if not rl1_pe_status and not rl1_eo_status:
                    print("Error in fairness computation vs expectation (rl1). Exiting.")
                    exit()
                
                rl2_pe_res, rl2_pe_status = rl_string_robustness(data2.values[i][9], epsilon, i, data2.values[i][1], data2.values[i][2], 3)
                rl2_eo_res, rl2_eo_status = rl_string_robustness(data2.values[i][9], epsilon, i, data2.values[i][1], data2.values[i][2], 4)
                rl2_eodds_comp = min(rl2_pe_res, rl2_eo_res)
                rl2_robustness.append(rl2_eodds_comp)
                if not rl2_pe_status and not rl2_eo_status:
                    print("Error in fairness computation vs expectation (rl2). Exiting.")
                    exit()

                rl3_pe_res, rl3_pe_status = rl_string_robustness(data3.values[i][9], epsilon, i, data3.values[i][1], data3.values[i][2], 3)
                rl3_eo_res, rl3_eo_status = rl_string_robustness(data3.values[i][9], epsilon, i, data3.values[i][1], data3.values[i][2], 4)
                rl3_eodds_comp = min(rl3_pe_res, rl3_eo_res)
                rl3_robustness.append(rl3_eodds_comp)
                if not rl3_pe_status and not rl3_eo_status:
                    print("Error in fairness computation vs expectation (rl3). Exiting.")
                    exit()

            else:
                print("unsupported metric. Exiting")
                exit()
        print("Jaccard distances (epsilon=", epsilon, "): ")
        for i in [0, 1, 2]:
            trivialModelsCount[i].append(rl_trivial_nb_cnt[i])
        average_rl1_robustness = np.average(rl1_robustness)
        average_rl2_robustness = np.average(rl2_robustness)
        average_rl3_robustness = np.average(rl3_robustness)
        print_detail = True
        if print_detail:
            print("0 masks: ", average_rl1_robustness, "(", rl1_robustness, ")")
            print("10 masks: ", average_rl2_robustness, "(", rl2_robustness, ")")
            print("30 masks: ", average_rl3_robustness, "(", rl3_robustness, ")")
            auditor.print_memo_statistics()
        else:
            print("0 masks: ", average_rl1_robustness)
            print("10 masks: ", average_rl2_robustness)
            print("30 masks: ", average_rl3_robustness)
        trainSampleRobustness[0].append(average_rl1_robustness)
        trainSampleRobustness[1].append(average_rl2_robustness)
        trainSampleRobustness[2].append(average_rl3_robustness)
    '''except:
        if debug>1:
            print("Missing results for epsilon=", epsilon, "(index=%d" %index, ") (metric=", fairnessMetric,") - files loaded :")
            print("File %d " %nb_masks1, ok1)
            print("File %d " %nb_masks2, ok2)
            print("File %d " %nb_masks3, ok3)
        tobererun.append(index)'''
    index += 1
    print("did ", 100*(index/len(epsilons)), "%.")

if debug>0:
    print("max cache size used = ", maxCache, "(", (maxCache/n_iter)*100, "%)")

if len(tobererun) > 0:
    print("---> Metric ", fairnessMetric, " indices to be re-run are : ", tobererun)
#exit()
with open('results_compiled/%s_DRO_compile_eps_metric%d_%s_sample_corrected_robustness_train_epsilon_solver_%s.csv' %(dataset, fairnessMetric, heuristic, solver_arg), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["epsilon", "train_sample_robustness_0masks", "train_sample_robustness_10masks", "train_sample_robustness_30masks", "trivial_nb_0masks", "trivial_nb_10masks", "trivial_nb_30masks" ])
    for index in range(len(epsilons)):
        csv_writer.writerow([epsilons[index], trainSampleRobustness[0][index], trainSampleRobustness[1][index], trainSampleRobustness[2][index], trivialModelsCount[0][index], trivialModelsCount[1][index], trivialModelsCount[2][index] ])

'''
with open('results_compiled/%s_DRO_compile_eps_metric%d_%s_unf_violation_%s.csv' %(dataset, fairnessMetric, heuristic, suffix), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["eps", "unf_train_%dmasks" %nb_masks1, "unf_train_%dmasks" %nb_masks2, "unf_train_%dmasks" %nb_masks3, "unf_test_%dmasks" %nb_masks1, "unf_test_%dmasks" %nb_masks2, "unf_test_%dmasks" %nb_masks3])
    for index in range(len(accTests[0])):
        csv_writer.writerow([effectiveEpsilons[index], unfTrains[0][index], unfTrains[1][index], unfTrains[2][index], unfTests[0][index], unfTests[1][index], unfTests[2][index]])


def clean_dominated(s1, s2, s3): # Removes repetitions and dominated solutions
    s1Tmp = []
    s2Tmp = []
    s3Tmp = []

    for i in range(len(s1)):
        isDominated = False
        repetition = False
        for j in range(len(s1)):
            if i != j:
                if ((s1[i]<s1[j]) and (s2[i]>=s2[j])) or ((s2[i]>s2[j]) and (s1[i]<=s1[j])):
                    isDominated = True
                    #print("[%lf,%lf] dominated by [%lf,%lf]\n" %(s1[i], s2[i], s1[j], s2[j]))
                elif (s1[i] == s1[j]) and (s2[i] == s2[j] and j < i):
                    repetition = True
        if(not isDominated and not repetition):
            s1Tmp.append(s1[i])
            s2Tmp.append(s2[i])
            s3Tmp.append(s3[i])


    return s1Tmp, s2Tmp, s3Tmp

epsilonsFinalTest = []
epsilonsFinalTrain = []

for i in range(3):
    epsilonsFinalTest.append(epsilons)
    epsilonsFinalTrain.append(epsilons)
    accTests[i], unfTests[i], epsilonsFinalTest[i] = clean_dominated(accTests[i], unfTests[i], epsilons)
    accTrains[i], unfTrains[i], epsilonsFinalTrain[i] = clean_dominated(accTrains[i], unfTrains[i], epsilons)


with open('results_compiled/%s_DRO_compile_eps_metric%d_%s_pareto_test_%s_%dmasks-withEpsilons.csv' %(dataset, fairnessMetric, heuristic, suffix, nb_masks1), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["epsilon", "acc_test", "unf_test"])
    for index in range(len(accTests[0])):
        csv_writer.writerow([epsilonsFinalTest[0][index], accTests[0][index], unfTests[0][index]])

exit(1)

with open('results_compiled/%s_DRO_compile_eps_metric%d_%s_pareto_test_%s_%dmasks.csv' %(dataset, fairnessMetric, heuristic, suffix, nb_masks1), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["acc_test", "unf_test"])
    for index in range(len(accTests[0])):
        csv_writer.writerow([accTests[0][index], unfTests[0][index]])

with open('results_compiled/%s_DRO_compile_eps_metric%d_%s_pareto_test_%s_%dmasks.csv' %(dataset, fairnessMetric, heuristic, suffix, nb_masks2), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["acc_test", "unf_test"])
    for index in range(len(accTests[1])):
        csv_writer.writerow([accTests[1][index], unfTests[1][index]])

with open('results_compiled/%s_DRO_compile_eps_metric%d_%s_pareto_test_%s_%dmasks.csv' %(dataset, fairnessMetric, heuristic, suffix, nb_masks3), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["acc_test", "unf_test"])
    for index in range(len(accTests[2])):
        csv_writer.writerow([accTests[2][index], unfTests[2][index]])

with open('results_compiled/%s_DRO_compile_eps_metric%d_%s_pareto_train_%s_%dmasks.csv' %(dataset, fairnessMetric, heuristic, suffix, nb_masks1), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["acc_train", "unf_train"])
    for index in range(len(accTrains[0])):
        csv_writer.writerow([accTrains[0][index], unfTrains[0][index]])

with open('results_compiled/%s_DRO_compile_eps_metric%d_%s_pareto_train_%s_%dmasks.csv' %(dataset, fairnessMetric, heuristic, suffix, nb_masks2), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["acc_train", "unf_train"])
    for index in range(len(accTrains[1])):
        csv_writer.writerow([accTrains[1][index], unfTrains[1][index]])

with open('results_compiled/%s_DRO_compile_eps_metric%d_%s_pareto_train_%s_%dmasks.csv' %(dataset, fairnessMetric, heuristic, suffix, nb_masks3), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["acc_train", "unf_train"])
    for index in range(len(accTrains[2])):
        csv_writer.writerow([accTrains[2][index], unfTrains[2][index]])

'''
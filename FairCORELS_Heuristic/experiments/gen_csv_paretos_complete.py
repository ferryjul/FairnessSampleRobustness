import pandas as pd 
import csv
import numpy as np
import argparse
from config import get_data, get_metric, get_strategy

parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 : statistical parity, 2 : predictive parity, 3 : predictive equality, 4 : equal opportunity, 5 : equalized odds, 6 : conditional use accuracy equality')
parser.add_argument('--dataset', type=int, default=1, help='Dataset. 1: Adult, 2: COMPAS, 3: Marketing, 4: Default Credit')
parser.add_argument('--filtering', type=int, default=0, help='Use improved CP filtering. 1: Yes, 0: No')

args = parser.parse_args()
dataset, decision, prediction_name, min_feature, min_pos, maj_feature, maj_pos = get_data(args.dataset)
suffix = "with_filtering" if args.filtering==1 else "without_filtering"

epsilon_range1 = np.arange(0.705, 0.901, 0.005)
epsilon_range2 = np.arange(0.902, 0.98, 0.002)
epsilon_range3 = np.arange(0.98, 0.9895, 0.001)
epsilon_range4 = np.arange(0.99, 1.000, 0.0002)
epsilon_range = list(epsilon_range1) + list(epsilon_range2) + list(epsilon_range3) + list(epsilon_range4)
base = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]
epsilon_range = base + epsilon_range
epsilons = [round(x,4) for x in epsilon_range] #147 values

fairnessMetric = args.metric#1

accTrains = [[], [], []]
accTests = [[], [], []]
unfTrains = [[], [], []]
unfTests = [[], [], []]

nb_masks1 = 0
nb_masks2 = 10
nb_masks3 = 30
heuristic = "BFS-objective-aware"
results_folder = "results-test"
effectiveEpsilons = []
index = 0
tobererun = []
debug = 1
maxCache = 0
n_iter = 25*10**5


for epsilon in epsilons:
    #epsilon = 1.0 - epsilonV
    try:
        ok1 = False
        ok2 = False
        ok3 = False
        data1 = pd.read_csv('./%s/%s_eps%f_metric%d_%s_%s_%dmasks.csv' %(results_folder, dataset, epsilon, fairnessMetric, heuristic, suffix, nb_masks1))
        ok1 = True
        data2 = pd.read_csv('./%s/%s_eps%f_metric%d_%s_%s_%dmasks.csv' %(results_folder, dataset, epsilon, fairnessMetric, heuristic, suffix, nb_masks2))
        ok2 = True
        data3 = pd.read_csv('./%s/%s_eps%f_metric%d_%s_%s_%dmasks.csv' %(results_folder, dataset, epsilon, fairnessMetric, heuristic, suffix, nb_masks3))
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

        if debug>0:
            for i in range(5):
                if data1.values[i][7] > maxCache:
                    maxCache = data1.values[i][7]
                if data2.values[i][7] > maxCache:
                    maxCache = data2.values[i][7]
                if data3.values[i][7] > maxCache:
                    maxCache = data3.values[i][7]
    except:
        if debug>1:
            print("Missing results for epsilon=", epsilon, "(index=%d" %index, ") (metric=", fairnessMetric,") - files loaded :")
            print("File %d " %nb_masks1, ok1)
            print("File %d " %nb_masks2, ok2)
            print("File %d " %nb_masks3, ok3)
        tobererun.append(index)
    index += 1

if debug>0:
    print("max cache size used = ", maxCache, "(", (maxCache/n_iter)*100, "%)")

if len(tobererun) > 0:
    print("---> Metric ", fairnessMetric, " indices to be re-run are : ", tobererun)
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

'''
with open('results_compiled/%s_DRO_compile_eps_metric%d_%s_pareto_test_%s_%dmasks-withEpsilons.csv' %(dataset, fairnessMetric, heuristic, suffix, nb_masks1), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["epsilon", "acc_test", "unf_test"])
    for index in range(len(accTests[0])):
        csv_writer.writerow([epsilonsFinalTest[0][index], accTests[0][index], unfTests[0][index]])

exit(1)
'''

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


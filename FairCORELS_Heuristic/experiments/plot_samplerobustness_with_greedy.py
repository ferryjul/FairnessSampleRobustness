import pandas as pd 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from config import get_data

parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 statistical_parity, 2 predictive_parity, 3 predictive_equality, 4 equal_opportunity')
parser.add_argument('--dataset', type=int, default=1, help='Dataset. 1: Adult, 2: COMPAS, 3: Marketing, 4: Default Credit')
parser.add_argument('--save', type=int, default=0, help='Show (0) or save (1) the Figure')
parser.add_argument('--fairness_inf', type=float, default=0.0, help='Only display fairness greater than this')
parser.add_argument('--fairness_sup', type=float, default=1.0, help='Only display fairness lower than this')
parser.add_argument('--eliminate_trivials', type=int, default=0, help='0 no 1 yes')
parser.add_argument('--show_trivials', type=int, default=0, help='0 no 1 yes')

args = parser.parse_args()
save_arg = args.save
fairnessMetric = args.metric
#dataset = args.dataset
dataset, decision, prediction_name, min_feature, min_pos, maj_feature, maj_pos = get_data(args.dataset)

heuristic = "BFS-objective-aware"

datasets=["adult", "compas"]

solvers = ['Mistral', 'OR-tools']
solver_arg = solvers[1]
fairnessMetricName=["Statistical Parity", "Predictive Parity", "Predictive Equality", "Equal Opportunity", "Equalized Odds"]

try:
    corr = "corrected_"
    fileName = 'results_compiled/%s_DRO_compile_eps_metric%d_%s_sample_%srobustness_train_epsilon_solver_%s.csv' %(dataset, fairnessMetric, heuristic, corr, solver_arg)
    fileContent = pd.read_csv(fileName)
    corr = "greedy_"
    fileNameHeuristic = 'results_compiled/%s_DRO_compile_eps_metric%d_%s_sample_%srobustness_train_epsilon_solver_%s.csv' %(dataset, fairnessMetric, heuristic, corr, solver_arg)
    fileContentHeuristic = pd.read_csv(fileNameHeuristic)
except FileNotFoundError:
    print("Missing results file for dataset %s, metric %s (%d)." %(dataset, fairnessMetricName[fairnessMetric-1], fairnessMetric))
    exit()


epsilonList=[]
sample_robustness_0_masksList=[]
sample_robustness_10_masksList=[]
sample_robustness_30_masksList=[]
sample_robustness_0_masksList_heuristic=[]
sample_robustness_10_masksList_heuristic=[]
sample_robustness_30_masksList_heuristic=[]
nb_trivial_0_masksList=[]
nb_trivial_10_masksList=[]
nb_trivial_30_masksList=[]
rowId = -1
fileContentHeuristicV = fileContentHeuristic.values
for row in fileContent.iterrows():
    rowId+=1
    epsilon=row[1][0]
    epsilonHeuristic=fileContentHeuristicV[rowId][0]
    if epsilon != epsilonHeuristic:
        print("error")
        exit()
    if epsilon < args.fairness_inf:
        continue
    if epsilon > args.fairness_sup:
        continue
    nb_trivial_0_masks=row[1][4]
    nb_trivial_10_masks=row[1][5]
    nb_trivial_30_masks=row[1][6]

    if fileContentHeuristicV[rowId][4] != nb_trivial_0_masks or fileContentHeuristicV[rowId][5] != nb_trivial_10_masks or fileContentHeuristicV[rowId][6] != nb_trivial_30_masks:
        print("error")
        exit()

    if args.eliminate_trivials > 0:
        if nb_trivial_0_masks > 0 or nb_trivial_10_masks > 0 or nb_trivial_30_masks > 0:
            continue
    sample_robustness_0_masks=row[1][1]
    sample_robustness_10_masks=row[1][2]
    sample_robustness_30_masks=row[1][3]
    
    sample_robustness_0_masks_heuristic=fileContentHeuristicV[rowId][1]
    sample_robustness_10_masks_heuristic=fileContentHeuristicV[rowId][2]
    sample_robustness_30_masks_heuristic=fileContentHeuristicV[rowId][3]
    epsilonList.append(epsilon)
    sample_robustness_0_masksList.append(sample_robustness_0_masks)
    sample_robustness_10_masksList.append(sample_robustness_10_masks)
    sample_robustness_30_masksList.append(sample_robustness_30_masks)

    sample_robustness_0_masksList_heuristic.append(sample_robustness_0_masks_heuristic)
    sample_robustness_10_masksList_heuristic.append(sample_robustness_10_masks_heuristic)
    sample_robustness_30_masksList_heuristic.append(sample_robustness_30_masks_heuristic)

    nb_trivial_0_masksList.append(nb_trivial_0_masks)
    nb_trivial_10_masksList.append(nb_trivial_10_masks)
    nb_trivial_30_masksList.append(nb_trivial_30_masks)

# plot sample robustness
colors = ['blue', 'orange', 'green']
fig,ax = plt.subplots()
'''ax.plot(epsilonList, sample_robustness_0_masksList, c=colors[0], label='no mask $\mathcal{IP-SR}(h,\mathcal{D},\epsilon)$')#, marker='o')
ax.plot(epsilonList, sample_robustness_10_masksList, c=colors[1], label='10 masks $\mathcal{IP-SR}(h,\mathcal{D},\epsilon)$')#, marker='o')
ax.plot(epsilonList, sample_robustness_30_masksList, c=colors[2], label='30 masks $\mathcal{IP-SR}(h,\mathcal{D},\epsilon)$')#, marker='o')
ax.plot(epsilonList, sample_robustness_0_masksList_heuristic, '--', c=colors[0],  label='no mask $\mathcal{G}reedy-\mathcal{SR}(h,\mathcal{D},\epsilon)$')#, marker='o')
ax.plot(epsilonList, sample_robustness_10_masksList_heuristic, '--', c=colors[1],  label='10 masks $\mathcal{G}reedy-\mathcal{SR}(h,\mathcal{D},\epsilon)$')#, marker='o')
ax.plot(epsilonList, sample_robustness_30_masksList_heuristic, '--', c=colors[2],  label='30 masks $\mathcal{G}reedy-\mathcal{SR}(h,\mathcal{D},\epsilon)$')#, marker='o')
ax.legend(loc='best')
'''
ax.plot(epsilonList, sample_robustness_0_masksList, c=colors[0])#, label='no mask $\mathcal{IP-SR}(h,\mathcal{D},\epsilon)$')#, marker='o')
ax.plot(epsilonList, sample_robustness_10_masksList, c=colors[1])#, label='10 masks $\mathcal{IP-SR}(h,\mathcal{D},\epsilon)$')#, marker='o')
ax.plot(epsilonList, sample_robustness_30_masksList, c=colors[2])#, label='30 masks $\mathcal{IP-SR}(h,\mathcal{D},\epsilon)$')#, marker='o')
ax.plot(epsilonList, sample_robustness_0_masksList_heuristic, '--', c=colors[0])#,  label='no mask $\mathcal{G}reedy-\mathcal{SR}(h,\mathcal{D},\epsilon)$')#, marker='o')
ax.plot(epsilonList, sample_robustness_10_masksList_heuristic, '--', c=colors[1])#,  label='10 masks $\mathcal{G}reedy-\mathcal{SR}(h,\mathcal{D},\epsilon)$')#, marker='o')
ax.plot(epsilonList, sample_robustness_30_masksList_heuristic, '--', c=colors[2])#,  label='30 masks $\mathcal{G}reedy-\mathcal{SR}(h,\mathcal{D},\epsilon)$')#, marker='o')
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], color=colors[0], lw=4, label='no mask'),
                    Line2D([0], [0], color=colors[1], lw=4, label='10 masks'),
                    Line2D([0], [0], color=colors[2], lw=4, label='30 masks'),
                    Line2D([0], [0], color='black', lw=1, label='$\mathcal{IPSR}(h,\mathcal{D},\epsilon)$'),
                    Line2D([0], [0], linestyle='--', color='black', lw=1, label='$\mathcal{G}reedy\mathcal{SR}(h,\mathcal{D},\epsilon)$')]                                                             

ax.legend(handles=legend_elements, loc='best')

ax.set_ylabel("Fairness sample-robustness")# (logarithmic scale)")
ax.set_xlabel("1-$\epsilon$")
#plt.yscale("log")

if args.show_trivials > 0:
    # add number of trivial models information
    ax2=ax.twinx()

    ax2.plot(epsilonList, nb_trivial_0_masksList, '--',  alpha=0.3, label='no mask', marker='x')
    ax2.plot(epsilonList, nb_trivial_10_masksList, '--',  alpha=0.3, label='10 masks', marker='x')
    ax2.plot(epsilonList, nb_trivial_30_masksList, '--',  alpha=0.3, label='30 masks', marker='x')
    ax2.set_ylabel("# trivial models")

if save_arg == 0:
    plt.show()
else:
    result_file_name = "Figures_SampleRobustness/%s_metric%d_samplerobustness_FunctionOf_epsilon_with_greedy.pdf" %(dataset, fairnessMetric)
    #plt.title("Sample-Robust Fairness, %s, %s dataset." %(fairnessMetricName[fairnessMetric-1], dataset))
    plt.title("%s metric" %(fairnessMetricName[fairnessMetric-1]))
    plt.savefig(result_file_name,bbox_inches='tight')
    print("Saved plot %s" %result_file_name)
import pandas as pd 
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

color_noval_all = 'grey'
color_noval_trivial  = 'red'
color_val_eps = 'skyblue'
color_val_train = 'darkblue'
color_val_all = 'darkviolet'
color_val_all_val = 'magenta'
colors_masks = {}
colors_masks[0] = 'gold' #'green'
colors_masks[10]= 'lime' #'magenta'
colors_masks[30] = 'darkgreen' #'gold'
color_epsilon = 'orange'
train_unf_marker = 'D'
test_unf_marker = 'o'
train_acc_marker = "v"
test_acc_marker = None#"x"

title = "dataset"
lw_general =  1#None # 1.0 #
lw_test = 1
markersize_value = 5

def find_position_log(aListofIndex, aListOfElements, anEl, m):
    index_correct = 0
    if anEl == aListOfElements[index_correct]:
        return 0
    while index_correct <= aListofIndex[-1] and anEl > aListOfElements[index_correct+1]:
        index_correct += 1
    if index_correct == 0 and anEl < aListOfElements[index_correct]:
        index_corrected = -0.5
    #print("Index_correct is ", index_correct, "anEl is ", anEl, "and aListOfElements[index_correct] is ", aListOfElements[index_correct],  ", aListOfElements[index_correct+1] is ",  aListOfElements[index_correct+1] )
    else:
        index_corrected = index_correct + abs((np.log10(anEl)-np.log10(aListOfElements[index_correct]))/(np.log10(aListOfElements[index_correct+1])-np.log10(aListOfElements[index_correct])))
    #print("index_corrected is ", index_corrected)
    return index_corrected#+(m/50)

def find_position(aListofIndex, aListOfElements, anEl, m):
    index_correct = 0
    while index_correct <= aListofIndex[-1] and anEl > aListOfElements[index_correct+1]:
        index_correct += 1
    #print("Index_correct is ", index_correct, "anEl is ", anEl, "and aListOfElements[index_correct] is ", aListOfElements[index_correct],  ", aListOfElements[index_correct+1] is ",  aListOfElements[index_correct+1] )
    index_corrected = index_correct + abs((anEl-aListOfElements[index_correct])/(aListOfElements[index_correct+1]-aListOfElements[index_correct]))
    #print("index_corrected is ", index_corrected)
    return index_corrected#+(m/50)

extension = "pdf"

fairnessMetricName=["Statistical Parity", "Predictive Parity", "Predictive Equality", "Equal Opportunity", "Equalized Odds"]
datasetNames = dict()
datasetNames["compas"]="Compas"
datasetNames["adult"]="Adult"
datasetNames["default_credit"]="Default of Credit Card Clients"
datasetNames["marketing"]="Bank Marketing"

parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 statistical_parity, 2 predictive_parity, 3 predictive_equality, 4 equal_opportunity')
parser.add_argument('--epsilon', type=float, default=0.0, help='epsilon value (min fairness acceptable) for epsilon-constrained method')
parser.add_argument('--dataset', type=str, default="compas", help='either adult or compas')
parser.add_argument('--plotType', type=int, default=3, help='type of plot to compute and display')
parser.add_argument('--show', type=int, default=0, help='Displays the plot if > 0. Saves it else.')

args = parser.parse_args()

useLB = 0 # using no filtering for these expes
epsilon = args.epsilon
fairnessMetric = args.metric
dataset=args.dataset
results_all = []
step = 0
fileFound = True
steps = []
plot_masks = True
includeValidation = True

showArg = args.show
# Read all results
while fileFound:
    try:
        fileName = './results_graham/results/sample_robust_frontier_%s_%d_faircorels_eps%f_metric%d_LB%d.csv' %(dataset, step, epsilon, fairnessMetric, useLB)
        fileContent = pd.read_csv(fileName)
        resPerFold = []
        for i in range(6):
            resPerFold.append({})
            resPerFold[i]["train_acc"] = fileContent.values[i][1]
            resPerFold[i]["train_unf"] = fileContent.values[i][2]
            resPerFold[i]["train_f_obj"] = fileContent.values[i][3]
            resPerFold[i]["train_sample_robustness"] = fileContent.values[i][4]
            resPerFold[i]["test_acc"] = fileContent.values[i][5]
            resPerFold[i]["test_unf"] = fileContent.values[i][6]
            if i == 5: # need to compute average
                resPerFold[i]["length"] = np.average(np.asarray([resPerFold[j]["length"] for j in range(5)]))
            else:
                resPerFold[i]["length"] = fileContent.values[i][9]
                resPerFold[i]["rule_list"] = fileContent.values[i][10]
        results_all.append(resPerFold)
        steps.append(step)
        step += 1
    except: 
        fileFound = False
        print("Found %d steps for epsilon=%f, dataset=%s, metric=%d" %(step, epsilon, dataset, fairnessMetric))

if step == 0:
    print("No result file found for epsilon=%f, dataset=%s, metric=%d. Exiting." %(epsilon, dataset, fairnessMetric))
    exit()

if plot_masks:
    ## Plot pts méthode masques
    # Retrieve results
    results_masks = dict()
    masks = [0, 10, 30]
    for m in masks:
        fileName='../../../FairCORELS_Heuristic/experiments/results-heuristic-sample_robust_fairness/%s_eps%f_metric%d_BFS-objective-aware_without_filtering_%dmasks.csv' %(dataset, epsilon, fairnessMetric, m)
        fileContent = pd.read_csv(fileName)
        resPerFold = []
        for i in range(6):
            resPerFold.append({})
            resPerFold[i]["train_acc"] = fileContent.values[i][1]
            resPerFold[i]["train_unf"] = fileContent.values[i][2]
            resPerFold[i]["train_f_obj"] = fileContent.values[i][3]
            resPerFold[i]["test_acc"] = fileContent.values[i][4]
            resPerFold[i]["test_unf"] = fileContent.values[i][5]
            if i == 5: # need to compute average
                resPerFold[i]["length"] = np.average(np.asarray([resPerFold[j]["length"] for j in range(5)]))
            else:
                resPerFold[i]["length"] = fileContent.values[i][8]
        results_masks[m] = resPerFold
    # Retrieve sample robustness audit results
    fileName='../../../FairCORELS_Heuristic/experiments/results_compiled/%s_DRO_compile_eps_metric%d_BFS-objective-aware_sample_corrected_robustness_train_epsilon_solver_OR-tools.csv' %(dataset, fairnessMetric)
    fileContent = pd.read_csv(fileName)
    i = 0
    while fileContent.values[i][0] != epsilon:
        i += 1
    results_masks[0][5]["train_sample_robustness"] = fileContent.values[i][1]
    results_masks[10][5]["train_sample_robustness"] = fileContent.values[i][2]
    results_masks[30][5]["train_sample_robustness"] = fileContent.values[i][3]


if includeValidation:
    steps_val = []
    results_all_val = []
    step_val = 0
    fileFound = True
    # Read all results
    while fileFound:
        try:
            fileName = './results_graham/results_with_validation/sample_robust_frontier_with-validation_%s_%d_faircorels_eps%f_metric%d_LB%d.csv' %(dataset, step_val, epsilon, fairnessMetric, useLB)
            fileContent = pd.read_csv(fileName)
            resPerFold = []
            for i in range(6):
                resPerFold.append({})
                resPerFold[i]["train_acc"] = fileContent.values[i][1]
                resPerFold[i]["train_unf"] = fileContent.values[i][2]
                resPerFold[i]["train_f_obj"] = fileContent.values[i][3]
                resPerFold[i]["train_sample_robustness"] = fileContent.values[i][4]
                resPerFold[i]["test_acc"] = fileContent.values[i][5]
                resPerFold[i]["test_unf"] = fileContent.values[i][6]
                resPerFold[i]["val_acc"] = fileContent.values[i][11]
                resPerFold[i]["val_unf"] = fileContent.values[i][12]
                if i == 5: # need to compute average
                    resPerFold[i]["length"] = np.average(np.asarray([resPerFold[j]["length"] for j in range(5)]))
                else:
                    resPerFold[i]["length"] = fileContent.values[i][9]
                    #resPerFold[i]["rule_list"] = fileContent.values[i][10]
            results_all_val.append(resPerFold)
            steps_val.append(step)
            step_val += 1
        except: 
            fileFound = False
            print("(validation results) Found %d steps for epsilon=%f, dataset=%s, metric=%d" %(step_val, epsilon, dataset, fairnessMetric))

    if step_val == 0:
        print("No result file found for epsilon=%f, dataset=%s, metric=%d. Exiting." %(epsilon, dataset, fairnessMetric))
        exit()
    else: # model selection step based on some criteria
        # Choice for the 5 folds simultaneously
        if args.plotType == 6 or args.plotType == 12 or args.plotType == 42 or args.plotType == 84 or args.plotType == 420 or args.plotType == 840 or args.plotType == 600 or args.plotType == 1200 or args.plotType == 601 or args.plotType == 1201:
            separateFolds = True
        elif args.plotType == 7 or args.plotType == 11 or  args.plotType == 14 or args.plotType == 22 or args.plotType == 17:
            separateFolds = True
        if not separateFolds:
            step_selection = 0
            while results_all_val[step_selection][5]["val_unf"] > 1.0 - epsilon:
                step_selection+=1
                if step_selection >= len(results_all_val)-1:
                    step_selection-=1
                    break
            folds_selection = [step_selection,step_selection,step_selection,step_selection,step_selection]
            
            step_selection = 0

            while results_all_val[step_selection][5]["val_unf"] > results_all_val[step_selection][5]["train_unf"]:
                step_selection+=1
                if step_selection >= len(results_all_val)-1:
                    step_selection-=1
                    break
            folds_selection_bis = [step_selection,step_selection,step_selection,step_selection,step_selection]

            # remove steps after the last validation point
            toremove = []
            for i in range(max([folds_selection[0], folds_selection_bis[0]]), len(results_all_val)-1):
                toremove.append(i)
            for i in toremove:
                del results_all_val[-1]

            step_selection = 0
            #print(len(results_all[-1][fold_id]["rule_list"].split("\n"))-1)
            while min([len(results_all[step_selection+1][fold_id]["rule_list"].split("\n"))-1 for fold_id in range(5)]) > 1:
                step_selection+=1
                if step_selection >= len(results_all)-1:
                    step_selection-=1
                    break
            folds_selection_ter = [step_selection,step_selection,step_selection,step_selection,step_selection]

            # remove steps after the last validation point
            '''toremove = []
            for i in range(folds_selection_ter[0], len(results_all)-1):
                toremove.append(i)
            for i in toremove:
                del results_all[-1]
            step = len(results_all)
            steps = [i for i in range(step)]'''
            #print("Model selected based on validation metrics is:", model_meets_epsilon_id)
        # Choice per fold
        else:
            folds_selection = []
            #folds_selection_vals = []
            for fold_id in range(5):
                step_selection = 0
                while results_all_val[step_selection][fold_id]["val_unf"] > 1.0 - epsilon:
                    step_selection+=1
                folds_selection.append(step_selection)
                #folds_selection_vals.append(results_all_val[step_selection][fold_id]["train_sample_robustness"])
            
            #print(folds_selection_vals)
            folds_selection_bis = []
            for fold_id in range(5):
                step_selection = 0
                while results_all_val[step_selection][fold_id]["val_unf"] > results_all_val[step_selection][fold_id]["train_unf"]:
                    step_selection+=1
                folds_selection_bis.append(step_selection)

            # remove steps after the last validation point
            toremove = []
            for i in range(max([max([folds_selection[i], folds_selection_bis[i]]) for i in range(5)]), len(results_all_val)-1):
                toremove.append(i)
            for i in toremove:
                del results_all_val[-1]

            folds_selection_ter = []
            for fold_id in range(5):
                step_selection = 0
                #print(len(results_all[-1][fold_id]["rule_list"].split("\n"))-1)
                while len(results_all[step_selection+1][fold_id]["rule_list"].split("\n"))-1 > 1:
                    step_selection+=1
                    if step_selection >= len(results_all)-1:
                        step_selection-=1
                        break
                #print("current: ",results_all[step_selection][fold_id]["rule_list"])
                #print("next: ", results_all[step_selection+1][fold_id]["rule_list"])
                if len(results_all[step_selection+1][fold_id]["rule_list"].split("\n"))-1 != 1:
                    print("error, next is not the constant classifier :(")
                folds_selection_ter.append(step_selection)
        
            # remove steps after the last validation point
            '''toremove = []
            for i in range(max([folds_selection_ter[i] for i in range(5)]), len(results_all)-1):
                toremove.append(i)
            for i in toremove:
                del results_all[-1]
            step = len(results_all)
            steps = [i for i in range(step)]'''
        print("Validation steps (epsilon criteria): ")
        print(folds_selection)
        print("Validation steps (training unfairness criteria): ")
        print(folds_selection_bis)
        print("No validation steps (before-trivial criteria): ")
        print(folds_selection_ter)

# Perform plots
plotType= args.plotType
print("Performing plot %d." %plotType)
## Overall fairness sample-robustness + overall fairness train AND test violation
if plotType == 6 or plotType == 12:

    #fig,ax = plt.subplots()
    overall_sample_robustness = [results_all[i][5]["train_sample_robustness"] for i in range(step)]
    #ax.plot(steps, overall_sample_robustness, '--', marker='o', color=color_noval_all, label="train fairness sample robustness")
    #ax.legend(loc='best')
    #ax.set_ylabel("Train Fairness Sample Robustness (log)")
    plt.xlabel("Steps for the Fairness Sample-Robust Frontier")
    logPlot = False
    if logPlot:
        plt.yscale("log")

    #ax2=ax.twinx()
    plt.ylabel("Unfairness")
    overall_unf_train = [results_all[i][5]["train_unf"] for i in range(step)]
    plt.plot(steps, overall_unf_train, color=color_noval_all, label="train unfairness", linewidth=lw_general)#, marker=train_unf_marker)
    overall_unf_test = [results_all[i][5]["test_unf"] for i in range(step)]
    plt.plot(steps, overall_unf_test, '--', color=color_noval_all, label="test unfairness", linewidth=lw_general)#, marker='o')
    #overall_unf_val = [results_all_val[i][5]["val_unf"] for i in range(len(results_all_val))]
    #ax2.plot([k for k in range(len(results_all_val))], overall_unf_val, marker='o', color='olive', label="validation unfairness")
    plt.plot([i for i in range(max([step, len(results_all_val)]))], [1.0-epsilon for i in range(max([step, len(results_all_val)]))], color=color_epsilon, label="epsilon", linewidth=lw_general)
    #ax2.legend(loc='best')
    if False:
        for m in masks:
            # Find good position for sample-robustness
            if logPlot:
                good_index = find_position_log(steps, overall_sample_robustness, results_masks[m][5]["train_sample_robustness"], m)
            else:
                good_index = find_position(steps, overall_sample_robustness, results_masks[m][5]["train_sample_robustness"], m)
            #ax.plot(good_index, results_masks[m][5]["train_sample_robustness"], color=colors_masks[m], marker='o', label='%d masks' %m)
            # Uses the same position for unfairness
            plt.plot(good_index, results_masks[m][5]["train_unf"], color=colors_masks[m], marker=train_unf_marker)
            plt.plot(good_index, results_masks[m][5]["test_unf"], color=colors_masks[m], marker=test_unf_marker)

    if includeValidation:
        # all data curves
        steps_val = [i for i in range(len(results_all_val))]
        plt.plot(steps_val, [results_all_val[i][5]["train_unf"] for i in steps_val], color=color_val_all, label="train unfairness", linewidth=lw_general)#, marker=train_unf_marker)
        plt.plot(steps_val, [results_all_val[i][5]["test_unf"] for i in steps_val], '--', color=color_val_all, label="test unfairness", linewidth=lw_general)#, marker=train_unf_marker)
        plt.plot(steps_val, [results_all_val[i][5]["val_unf"] for i in steps_val], ':', color=color_val_all_val, label="validation unfairness", linewidth=lw_general)#, marker=train_unf_marker)

        # epsilon criteria
        sample_r_model_meets_epsilon_sample_r = np.average([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection)])
        sample_r_model_meets_epsilon_unf = np.average([results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection)])
        sample_r_model_meets_epsilon_unf_train = np.average([results_all_val[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection)])
        if logPlot:
            good_index = find_position_log(steps, overall_sample_robustness, sample_r_model_meets_epsilon_sample_r, -1)
        else:
            #good_index = find_position(steps, overall_sample_robustness, sample_r_model_meets_epsilon_sample_r, -1) # last parameter not used anymore
            good_index = np.average(folds_selection)
        #ax.plot(good_index, sample_r_model_meets_epsilon_sample_r, marker='o', color=color_val_eps, label='val (epsilon criteria)')
        # Uses the same position for unfairness
        plt.plot(good_index, sample_r_model_meets_epsilon_unf_train, marker=train_unf_marker, color=color_val_eps)
        plt.plot(good_index, sample_r_model_meets_epsilon_unf, marker=test_unf_marker, color=color_val_eps)

        # train fairness criteria
        sample_r_model_meets_epsilon_sample_r_bis = np.average([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_bis)])
        sample_r_model_meets_epsilon_unf_bis = np.average([results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_bis)])
        sample_r_model_meets_epsilon_unf_train_bis = np.average([results_all_val[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection_bis)])
        if logPlot:
            good_index_bis = find_position_log(steps, overall_sample_robustness, sample_r_model_meets_epsilon_sample_r_bis, -1)
        else:
            #good_index_bis = find_position(steps, overall_sample_robustness, sample_r_model_meets_epsilon_sample_r_bis, -1) # last parameter not used anymore
            good_index_bis = np.average(folds_selection_bis)
        #ax.plot(good_index_bis, sample_r_model_meets_epsilon_sample_r_bis, marker='o', color=color_val_train, label='val (train unf criteria)')
        
        plt.plot(good_index_bis, sample_r_model_meets_epsilon_unf_train_bis, marker=train_unf_marker, color=color_val_train)
        plt.plot(good_index_bis, sample_r_model_meets_epsilon_unf_bis, marker=test_unf_marker, color=color_val_train)

        # before constant classifier criteria
        before_constant_sample_r = np.average([results_all[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_ter)])
        before_constant_unf = np.average([results_all[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_ter)])
        before_constant_unf_train = np.average([results_all[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection_ter)])
        if logPlot:
            good_index_ter = find_position_log(steps, overall_sample_robustness, before_constant_sample_r, -1)
        else:
            #good_index_ter = find_position(steps, overall_sample_robustness, before_constant_sample_r, -1) # last parameter not used anymore
            good_index_ter = np.average(folds_selection_ter)
        #ax.plot(good_index_ter, before_constant_sample_r, marker='o', color='black', label='noval (before-constant criteria)')

        # Uses the same position for unfairness
        plt.plot(good_index_ter, before_constant_unf_train, marker=train_unf_marker, color=color_noval_trivial)
        plt.plot(good_index_ter, before_constant_unf, marker=test_unf_marker, color=color_noval_trivial)

    if plotType == 12:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legendFig = plt.figure("Legend plot")
        #ax = fig.add_subplot(111)
        legend_elements = [
                    Line2D([0], [0], color=color_noval_all, lw=4, label='sample robust fair frontier (no validation)'),
                    Line2D([0], [0], color=color_noval_trivial, lw=4, label='no validation (before-constant)'),
                    Line2D([0], [0], color=color_val_all, lw=4, label='sample robust fair frontier (validation)'),
                    Line2D([0], [0], color=color_val_eps, lw=4,  label='validation ($\epsilon$ criterion)'),
                    Line2D([0], [0], color=color_val_train, lw=4, label='validation (train unf. criterion)'),
                    Line2D([0], [0], color=color_epsilon, lw=1, label='$\epsilon$'),
                    Line2D([0], [0], marker=None, color=color_val_all_val, lw=1, linestyle = ':', label='validation unfairness'),
                    Line2D([0], [0], marker=train_unf_marker, color='black', lw=1, label='train unfairness'),
                    Line2D([0], [0], marker=test_unf_marker, color='black', lw=1,  linestyle = '--', label='test unfairness')]     
                    #Line2D([0], [0], color=colors_masks[0], lw=4, label='no mask'),
                    #Line2D([0], [0], color=colors_masks[10], lw=4, label='10 masks'),
                    #Line2D([0], [0], color=colors_masks[30], lw=4, label='30 masks'),

        '''
        color_noval_all = 'grey'
        color_noval_trivial  = 'red'
        color_val_eps = 'skyblue'
        color_val_train = 'darkblue'
        color_val_all = 'darkviolet'
        color_val_all_val = 'magenta'
        colors_masks = {}
        colors_masks[0] = 'gold' #'green'
        colors_masks[10]= 'lime' #'magenta'
        colors_masks[30] = 'darkgreen' #'gold'
        color_epsilon = 'orange'
        train_unf_marker = 'D'
        test_unf_marker = 'o'
        '''                                                        

        legendFig.legend(handles=legend_elements, loc='center', ncol=3)
        if showArg>0:
            legendFig.show()
        else:
            legendFig.savefig('./Paretos_All_Figures/sample-robustness-steps_legend.pdf', bbox_inches='tight')
        exit()
        '''else:
        handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]

        handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
        fig.legend(handles, labels, loc="lower center",   # Position of legend
            borderaxespad=0.1, ncol=3) #ncol=2)'''
    #fig.subplots_adjust(bottom=0.2)
    #plt.title("Dataset %s, metric %s, $\epsilon=%f$" %(datasetNames[dataset], fairnessMetricName[fairnessMetric-1], epsilon))
    if title=="metric":
        plt.title("%s metric" %(fairnessMetricName[fairnessMetric-1]))
    elif title=="dataset":
        plt.title("%s dataset" %(datasetNames[dataset]))
    if showArg>0:
        plt.show()
    else:
        plt.savefig("./Paretos_All_Figures/sample-robustness-steps_dataset_%s-metric_%d-epsilon_%f.%s" %(dataset, fairnessMetric, epsilon, extension), bbox_inches='tight')
# Sample-robustness 1D
elif plotType == 7 or plotType == 14:

    fig1 = plt.figure(figsize=(7, 2), facecolor='white', tight_layout=True)
    ax1 = plt.axes(frameon=False)
    ax1.get_xaxis().tick_bottom()
    ax1.margins(x=0.01, tight=True)
    overall_sample_robustness = [results_all[i][5]["train_sample_robustness"] for i in range(step)]
    #ax.plot(steps, overall_sample_robustness, '--', marker='o', color=color_noval_all, label="train fairness sample robustness")
    #ax.legend(loc='best')
    #ax.set_ylabel("Train Fairness Sample Robustness (log)")
    ax1.set_xlabel("Sample-Robustness")
    #ax1.get_yaxis().set_visible(False)
    logPlot = True
    if logPlot:
        plt.xscale("log")
    ax1.plot([0,1],[0,0], color="black",linewidth=0.5)
    ax1.plot([0,1],[1,1], color="black",linewidth=0.5)
    ax1.plot([0,1],[2,2], color="black",linewidth=0.5)
    #ax2=ax.twinx()
    #plt.ylabel("Unfairness")
    overall_unf_train = [results_all[i][5]["train_unf"] for i in range(step)]
    
    #plt.plot(steps, overall_unf_train, color=color_noval_all, label="train unfairness")#, marker=train_unf_marker)
    overall_unf_test = [results_all[i][5]["test_unf"] for i in range(step)]
    #plt.plot(steps, overall_unf_test, '--', color=color_noval_all, label="test unfairness")#, marker='o')
    #overall_unf_val = [results_all_val[i][5]["val_unf"] for i in range(len(results_all_val))]
    #ax2.plot([k for k in range(len(results_all_val))], overall_unf_val, marker='o', color='olive', label="validation unfairness")
    #plt.plot(steps, [1.0-epsilon for i in steps], color=color_epsilon, label="epsilon")
    #ax2.legend(loc='best')
    if plot_masks:
        for m in masks:
            # Find good position for sample-robustness
            if logPlot:
                good_index = find_position_log(steps, overall_sample_robustness, results_masks[m][5]["train_sample_robustness"], m)
            else:
                good_index = find_position(steps, overall_sample_robustness, results_masks[m][5]["train_sample_robustness"], m)
            #ax.plot(good_index, results_masks[m][5]["train_sample_robustness"], color=colors_masks[m], marker='o', label='%d masks' %m)
            # Uses the same position for unfairness
            #plt.plot(good_index, results_masks[m][5]["train_unf"], color=colors_masks[m], marker=train_unf_marker)
            #plt.plot(good_index, results_masks[m][5]["test_unf"], color=colors_masks[m], marker='o')
            ax1.scatter(results_masks[m][5]["train_sample_robustness"], 2, marker='o', color=colors_masks[m])

    if includeValidation:
        # all data curves
        steps_val = [i for i in range(len(results_all_val))]
        #plt.plot(steps_val, [results_all_val[i][5]["train_unf"] for i in steps_val], color='blue', label="train unfairness")#, marker=train_unf_marker)
        #plt.plot(steps_val, [results_all_val[i][5]["test_unf"] for i in steps_val], '--', color='blue', label="test unfairness")#, marker=train_unf_marker)
       
        # epsilon criteria
        sample_r_model_meets_epsilon_sample_r = np.average([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection)])
        sample_r_model_meets_epsilon_unf = np.average([results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection)])
        sample_r_model_meets_epsilon_unf_train = np.average([results_all_val[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection)])
        
        #ax.plot(good_index, sample_r_model_meets_epsilon_sample_r, marker='o', color=color_val_eps, label='val (epsilon criteria)')
        # Uses the same position for unfairness
        #plt.plot(good_index, sample_r_model_meets_epsilon_unf_train, marker=train_unf_marker, color=color_val_eps)
        #plt.plot(good_index, sample_r_model_meets_epsilon_unf, marker='o', color=color_val_eps)
        ax1.scatter(sample_r_model_meets_epsilon_sample_r, 1, marker='o', color=color_val_eps)
        # train fairness criteria
        sample_r_model_meets_epsilon_sample_r_bis = np.average([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_bis)])
        sample_r_model_meets_epsilon_unf_bis = np.average([results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_bis)])
        sample_r_model_meets_epsilon_unf_train_bis = np.average([results_all_val[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection_bis)])
        
        #ax.plot(good_index_bis, sample_r_model_meets_epsilon_sample_r_bis, marker='o', color=color_val_train, label='val (train unf criteria)')

        #plt.plot(good_index_bis, sample_r_model_meets_epsilon_unf_train_bis, marker=train_unf_marker, color=color_val_train)
        #plt.plot(good_index_bis, sample_r_model_meets_epsilon_unf_bis, marker='o', color=color_val_train)
        ax1.scatter(sample_r_model_meets_epsilon_sample_r_bis, 1, marker='o', color=color_val_train)

        # before constant classifier criteria
        before_constant_sample_r = np.average([results_all[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_ter)])
        before_constant_unf = np.average([results_all[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_ter)])
        before_constant_unf_train = np.average([results_all[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection_ter)])
        
        #ax.plot(good_index_ter, before_constant_sample_r, marker='o', color='black', label='noval (before-constant criteria)')
        ax1.scatter(before_constant_sample_r, 0, marker='o', color=color_noval_trivial)
        # Uses the same position for unfairness
        #plt.plot(good_index_ter, before_constant_unf_train, marker=train_unf_marker, color='black')
        #plt.plot(good_index_ter, before_constant_unf, marker='o', color='black')
    ax1.scatter(overall_sample_robustness, [0 for i in range(len(steps))], marker='|', color=color_noval_all)#'black')
    ax1.scatter(overall_sample_robustness, [-1 for i in range(len(steps))], marker='')
    ax1.scatter(overall_sample_robustness, [3 for i in range(len(steps))], marker='')
    y1 = [0, 1, 2]
    squad = ['Exact','Exact + Val','Heuristic']#'Masks']
    ax1.set_yticks(y1)
    ax1.set_yticklabels(squad, minor=False, rotation=45)
    if plotType == 14:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legendFig = plt.figure("Legend plot")
        legend_elements = [Line2D([0], [0], marker='o', color=colors_masks[0], lw=4, linestyle = 'None', label='no mask'),
                    Line2D([0], [0], marker='o', color=colors_masks[10], lw=4, linestyle = 'None',label='10 masks'),
                    Line2D([0], [0], marker='o', color=colors_masks[30], lw=4, linestyle = 'None',label='30 masks'),
                    Line2D([0], [0], marker='o', color=color_val_eps, lw=4, linestyle = 'None',  label='validation ($\epsilon$ criterion)'),
                    Line2D([0], [0], marker='o', color=color_val_train, lw=4, linestyle = 'None', label='validation (train unf. criterion)'),
                    Line2D([0], [0], marker='|', color=color_noval_all, lw=4, markersize=10, linestyle = 'None', label='sample robust fair frontier (no validation)'),
                    Line2D([0], [0], marker='o', color=color_noval_trivial, lw=4, linestyle = 'None', label='no validation (before-constant)')]                           
                    #Line2D([0], [0], color=colors_masks[0], lw=4, label='no mask'),
                    #Line2D([0], [0], color=colors_masks[10], lw=4, label='10 masks'),
                    #Line2D([0], [0], color=colors_masks[30], lw=4, label='30 masks'),
                                       

        legendFig.legend(handles=legend_elements, loc='center', ncol=3)
        legendFig.savefig('./Paretos_All_Figures/zzz_figures_exact_faircorels_sample-robustness_legend.pdf', bbox_inches='tight')
        exit()
    '''else:
        handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]

        handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
        fig.legend(handles, labels, loc="lower center",   # Position of legend
            borderaxespad=0.1, ncol=3) #ncol=2)'''
    #fig.subplots_adjust(bottom=0.2)
    #plt.title("Dataset %s, metric %s, $\epsilon=%f$" %(datasetNames[dataset], fairnessMetricName[fairnessMetric-1], epsilon))
    if title=="metric":
        plt.title("%s metric" %(fairnessMetricName[fairnessMetric-1]))
    elif title=="dataset":
        plt.title("%s dataset" %(datasetNames[dataset]))
    if showArg>0:
        plt.show()
    else:
        plt.savefig("./Paretos_All_Figures/zzz_figures_exact_faircorels_sample-robustness_%s-metric_%d-epsilon_%f.%s" %(dataset, fairnessMetric, epsilon, extension), bbox_inches='tight')
##  overall test unfairness function of overall test error
elif plotType == 11 or plotType == 22:
    #plt.figure(figsize=(7.0, 4.5))# default value: (6.4, 4.8))
    if plot_masks:
        for m in masks:
            plt.scatter(1.0-results_masks[m][5]["test_acc"], results_masks[m][5]["test_unf"], color=colors_masks[m], marker='o', label='%d masks' %m)
            #plt.scatter(1.0-results_masks[m][5]["test_acc"], results_masks[m][5]["test_unf"], facecolors='none', edgecolors=colors_masks[m], label='%d masks' %m)

    overall_sample_robustness = [1.0-results_all[i][5]["test_acc"] for i in range(step)]
    overall_unf_test = [results_all[i][5]["test_unf"] for i in range(step)]
    plt.scatter(overall_sample_robustness, overall_unf_test, marker='1', color=color_noval_all) #"."
    #plt.xscale("log")
    #plt.legend(loc='upper center')
    plt.xlabel("Test Error")
    plt.ylabel("Test Unfairness")

    if includeValidation:
        # all folds simultaneously
        # epsilon criteria
        sample_r_model_meets_epsilon_acc = np.average([1.0-results_all_val[modelS][fold]["test_acc"] for fold, modelS in enumerate(folds_selection)])
        sample_r_model_meets_epsilon_unf = np.average([results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection)])
        plt.scatter(sample_r_model_meets_epsilon_acc, sample_r_model_meets_epsilon_unf, marker='o', color=color_val_eps, label='val (epsilon criteria)')
        # train unf criteria
        sample_r_model_meets_epsilon_acc_bis = np.average([1.0-results_all_val[modelS][fold]["test_acc"] for fold, modelS in enumerate(folds_selection_bis)])
        sample_r_model_meets_epsilon_unf_bis = np.average([results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_bis)])
        plt.scatter(sample_r_model_meets_epsilon_acc_bis, sample_r_model_meets_epsilon_unf_bis, marker='o', color=color_val_train, label='val (train unf criteria)')

        # before constant classifier criteria
        before_constant_unf = np.average([results_all[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_ter)])
        before_constant_acc = np.average([1.0 - results_all[modelS][fold]["test_acc"] for fold, modelS in enumerate(folds_selection_ter)])  
        plt.scatter(before_constant_acc, before_constant_unf, marker='o', color=color_noval_trivial, label='noval (before constant criteria)')

    if plotType == 22:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legendFig = plt.figure("Legend plot")
        legend_elements = [
                    Line2D([0], [0], marker='o', color=colors_masks[0], lw=4, linestyle = 'None', label='no mask'),
                    Line2D([0], [0], marker='o', color=colors_masks[10], lw=4, linestyle = 'None',label='10 masks'),
                    Line2D([0], [0], marker='o', color=colors_masks[30], lw=4, linestyle = 'None',label='30 masks'),
                    Line2D([0], [0], marker='o', color=color_val_eps, lw=4, linestyle = 'None',  label='validation ($\epsilon$ criterion)'),
                    Line2D([0], [0], marker='o', color=color_val_train, lw=4, linestyle = 'None', label='validation (train unf. criterion)'),
                    Line2D([0], [0], color=color_epsilon, lw=1, label='$\epsilon$'),
                    Line2D([0], [0], marker='1', color=color_noval_all, lw=4, linestyle = 'None', label='sample robust fair frontier (no validation)'),
                    Line2D([0], [0], marker='o', color=color_noval_trivial, lw=4, linestyle = 'None', label='no validation (before-constant)')]                                                             

        legendFig.legend(handles=legend_elements, loc='center', ncol=3)
        legendFig.savefig('./Paretos_All_Figures/test-unfairness-function_of-test-error_legend.pdf', bbox_inches='tight')
        exit()

    plt.plot(overall_sample_robustness+[sample_r_model_meets_epsilon_acc,sample_r_model_meets_epsilon_acc_bis,before_constant_acc], [1.0-epsilon for i in range(len(steps)+3)], color=color_epsilon, label="epsilon")
    #plt.title("Dataset %s, metric %d, epsilon=%f" %(dataset, fairnessMetric, epsilon))
    #plt.title("Dataset %s, metric %s, $\epsilon=%f$" %(datasetNames[dataset], fairnessMetricName[fairnessMetric-1], epsilon))
    if title=="metric":
        plt.title("%s metric" %(fairnessMetricName[fairnessMetric-1]))
    elif title=="dataset":
        plt.title("%s dataset" %(datasetNames[dataset]))
    #plt.legend(loc='best')
    if showArg>0:
        plt.show()
    else:
        #plt.savefig("./test-unfairness-function_of-test-error-dataset_%s-metric_%d-epsilon_%f.%s" %(dataset, fairnessMetric, epsilon, extension), bbox_inches='tight')
        plt.savefig("./Paretos_All_Figures/test-unfairness-function_of-test-error-dataset_%s-metric_%d-epsilon_%f.%s" %(dataset, fairnessMetric, epsilon, extension), bbox_inches='tight')
# Save test information in csv
elif plotType == 17:
    print("done")
    if plot_masks:
        for m in masks:
            plt.scatter(1.0-results_masks[m][5]["test_acc"], results_masks[m][5]["test_unf"], color=colors_masks[m], marker='o', label='%d masks' %m)
            #plt.scatter(1.0-results_masks[m][5]["test_acc"], results_masks[m][5]["test_unf"], facecolors='none', edgecolors=colors_masks[m], label='%d masks' %m)
    
    with open('./results/results-metric_%d-epsilon_%f-dataset_%s.csv' %(args.metric, args.epsilon, args.dataset), mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header_row = ['epsilon', 'value']#'epsilon']
        for m in masks:
            header_row.append("%d_masks" %m)
        header_row = header_row + ['before-constant', 'val_epsilon_criterion', 'val_train_unf_criterion']
        csv_writer.writerow(header_row)

        results_row = [args.epsilon, "test_error"]
        for m in masks:
            results_row.append(1.0 - results_masks[m][5]["test_acc"])
        results_row.append(np.average([1.0 - results_all[modelS][fold]["test_acc"] for fold, modelS in enumerate(folds_selection_ter)]))
        results_row.append(np.average([1.0 - results_all_val[modelS][fold]["test_acc"] for fold, modelS in enumerate(folds_selection)]))
        results_row.append(np.average([1.0 - results_all_val[modelS][fold]["test_acc"] for fold, modelS in enumerate(folds_selection_bis)]))
        csv_writer.writerow(results_row)

        results_row = [args.epsilon, "test_unf"]
        for m in masks:
            results_row.append(results_masks[m][5]["test_unf"])
        results_row.append(np.average([results_all[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_ter)]))
        results_row.append(np.average([results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection)]))
        results_row.append(np.average([results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_bis)]))
        csv_writer.writerow(results_row)

# Train sample-robustness and train accuracy
elif plotType == 42 or plotType == 84:
    fig,ax = plt.subplots()
    overall_sample_robustness = [results_all[i][5]["train_sample_robustness"] for i in range(step)]
    ax.plot(steps, overall_sample_robustness, '--', marker='o', color=color_noval_all, label="train fairness sample robustness")
    #ax.legend(loc='best')
    ax.set_ylabel("Train Fairness Sample Robustness (log)")
    plt.xlabel("Steps for the Fairness Sample-Robust Frontier")
    logPlot = True
    if logPlot:
        ax.set_yscale("log")

    ax2=ax.twinx()
    plt.ylabel("Train Accuracy")
    overall_err_train = [results_all[i][5]["train_acc"] for i in range(step)]
    ax2.plot(steps, overall_err_train, color=color_noval_all, marker=train_acc_marker, label="train unfairness", linewidth=lw_general)#, marker=train_unf_marker)
    if False:
        for m in masks:
            # Find good position for sample-robustness
            if logPlot:
                good_index = find_position_log(steps, overall_sample_robustness, results_masks[m][5]["train_sample_robustness"], m)
            else:
                good_index = find_position(steps, overall_sample_robustness, results_masks[m][5]["train_sample_robustness"], m)
            ax.plot(good_index, results_masks[m][5]["train_sample_robustness"], color=colors_masks[m], marker='o', label='%d masks' %m)
            # Uses the same position for unfairness
            #ax2.scatter(good_index, results_masks[m][5]["train_sample_robustness"], color=colors_masks[m], marker=train_unf_marker)
            #ax2.scatter(good_index, results_masks[m][5]["train_acc"], color=colors_masks[m], marker=test_unf_marker)

    if includeValidation:
        # all data curves
        steps_val = [i for i in range(len(results_all_val))]
        overall_sample_robustness_val = [results_all_val[i][5]["train_sample_robustness"] for i in steps_val]
        ax.plot(steps_val, overall_sample_robustness_val, marker='o', color=color_val_all, label="validation", linewidth=lw_general)#, marker=train_unf_marker)
        ax2.plot(steps_val, [results_all_val[i][5]["train_acc"] for i in steps_val], '--', marker=train_acc_marker, color=color_val_all, label="validation", linewidth=lw_general)#, marker=train_unf_marker)
        #plt.plot(steps_val, [results_all_val[i][5]["val_unf"] for i in steps_val], ':', color=color_val_all_val, label="validation unfairness", linewidth=lw_general)#, marker=train_unf_marker)

        # epsilon criteria
        sample_r_model_meets_epsilon_sample_r = np.average([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection)])
        sample_r_model_meets_epsilon_acc = np.average([results_all_val[modelS][fold]["train_acc"] for fold, modelS in enumerate(folds_selection)])
        if logPlot:
            good_index = find_position_log(steps, overall_sample_robustness_val, sample_r_model_meets_epsilon_sample_r, -1)
        else:
            good_index = find_position(steps, overall_sample_robustness, sample_r_model_meets_epsilon_sample_r, -1) # last parameter not used anymore
        ax.plot(good_index, sample_r_model_meets_epsilon_sample_r, marker='o', color=color_val_eps, label='val (epsilon criteria)')
        ax2.plot(good_index, sample_r_model_meets_epsilon_acc, marker=train_acc_marker, color=color_val_eps, label='val (train unf criteria)')

        # train fairness criteria
        sample_r_model_meets_epsilon_sample_r_bis = np.average([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_bis)])
        sample_r_model_meets_epsilon_acc_bis = np.average([results_all_val[modelS][fold]["train_acc"] for fold, modelS in enumerate(folds_selection_bis)])
        if logPlot:
            good_index_bis = find_position_log(steps, overall_sample_robustness_val, sample_r_model_meets_epsilon_sample_r_bis, -1)
        else:
            good_index_bis = find_position(steps, overall_sample_robustness, sample_r_model_meets_epsilon_sample_r_bis, -1) # last parameter not used anymore
        ax.plot(good_index_bis, sample_r_model_meets_epsilon_sample_r_bis, marker='o', color=color_val_train, label='val (train unf criteria)')
        ax2.plot(good_index_bis, sample_r_model_meets_epsilon_acc_bis, marker=train_acc_marker, color=color_val_train, label='val (train unf criteria)')

        # before constant classifier criteria
        before_constant_sample_r = np.average([results_all[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_ter)])
        before_constant_acc = np.average([results_all[modelS][fold]["train_acc"] for fold, modelS in enumerate(folds_selection_ter)])
        if logPlot:
            good_index_ter = find_position_log(steps, overall_sample_robustness, before_constant_sample_r, -1)
        else:
            good_index_ter = find_position(steps, overall_sample_robustness, before_constant_sample_r, -1) # last parameter not used anymore
        ax.plot(good_index_ter, before_constant_sample_r, marker='o', color=color_noval_trivial, label='noval (before-constant criteria)')
        ax2.plot(good_index_ter, before_constant_acc, marker=train_acc_marker, color=color_noval_trivial, label='noval (before-constant criteria)')

        # Uses the same position for unfairness
        #plt.plot(good_index_ter, before_constant_unf_train, marker=train_unf_marker, color=color_noval_trivial)
        #plt.plot(good_index_ter, before_constant_unf, marker=test_unf_marker, color=color_noval_trivial)

# Train sample-robustness and train accuracy, minor revision -> INCLUDE VALIDATION OR TEST DATA ACCURACY
elif plotType == 420 or plotType == 840:
    fig,ax = plt.subplots()
    overall_sample_robustness = [results_all[i][5]["train_sample_robustness"] for i in range(step)]
    ax.plot(steps, overall_sample_robustness, marker='o', markersize=markersize_value, color=color_noval_all, label="train fairness sample robustness")
    #ax.legend(loc='best')
    ax.set_ylabel("Train Fairness Sample Robustness (log)")
    plt.xlabel("Steps for the Fairness Sample-Robust Frontier")
    logPlot = True
    if logPlot:
        ax.set_yscale("log")

    ax2=ax.twinx()
    plt.ylabel("Accuracy (test and train)")
    overall_err_train = [results_all[i][5]["train_acc"] for i in range(step)]
    overall_err_test = [results_all[i][5]["test_acc"] for i in range(step)]
    ax2.plot(steps, overall_err_train, color=color_noval_all, marker=train_acc_marker, markersize=markersize_value, label="train unfairness", linewidth=lw_general)#, marker=train_unf_marker)
    # Below: ajout minor revision
    ax2.plot(steps, overall_err_test, '--', color=color_noval_all, marker=test_acc_marker, markersize=markersize_value, label="train unfairness", linewidth=lw_test)#, marker=train_unf_marker)
    if False:
        for m in masks:
            # Find good position for sample-robustness
            if logPlot:
                good_index = find_position_log(steps, overall_sample_robustness, results_masks[m][5]["train_sample_robustness"], m)
            else:
                good_index = find_position(steps, overall_sample_robustness, results_masks[m][5]["train_sample_robustness"], m)
            ax.plot(good_index, results_masks[m][5]["train_sample_robustness"], color=colors_masks[m], marker='o', label='%d masks' %m)
            # Uses the same position for unfairness
            #ax2.scatter(good_index, results_masks[m][5]["train_sample_robustness"], color=colors_masks[m], marker=train_unf_marker)
            #ax2.scatter(good_index, results_masks[m][5]["train_acc"], color=colors_masks[m], marker=test_unf_marker)

    if includeValidation:
        # all data curves
        steps_val = [i for i in range(len(results_all_val))]
        overall_sample_robustness_val = [results_all_val[i][5]["train_sample_robustness"] for i in steps_val]
        ax.plot(steps_val, overall_sample_robustness_val, marker='o', markersize=markersize_value, color=color_val_all, label="validation", linewidth=lw_general)#, marker=train_unf_marker)
        ax2.plot(steps_val, [results_all_val[i][5]["train_acc"] for i in steps_val], marker=train_acc_marker, markersize=markersize_value, color=color_val_all, label="validation", linewidth=lw_general)#, marker=train_unf_marker)
        #plt.plot(steps_val, [results_all_val[i][5]["val_unf"] for i in steps_val], ':', color=color_val_all_val, label="validation unfairness", linewidth=lw_general)#, marker=train_unf_marker)
        # Below: ajout minor revision
        ax2.plot(steps_val, [results_all_val[i][5]["test_acc"] for i in steps_val], '--', marker=test_acc_marker, markersize=markersize_value, color=color_val_all, label="validation", linewidth=lw_test)#, marker=train_unf_marker)

        # epsilon criteria
        sample_r_model_meets_epsilon_sample_r = np.average([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection)])
        sample_r_model_meets_epsilon_acc = np.average([results_all_val[modelS][fold]["train_acc"] for fold, modelS in enumerate(folds_selection)])
        if logPlot:
            good_index = find_position_log(steps, overall_sample_robustness_val, sample_r_model_meets_epsilon_sample_r, -1)
        else:
            good_index = find_position(steps, overall_sample_robustness, sample_r_model_meets_epsilon_sample_r, -1) # last parameter not used anymore
        ax.plot(good_index, sample_r_model_meets_epsilon_sample_r, marker='o', color=color_val_eps, label='val (epsilon criteria)')
        ax2.plot(good_index, sample_r_model_meets_epsilon_acc, marker=train_acc_marker, color=color_val_eps, label='val (train unf criteria)')

        # train fairness criteria
        sample_r_model_meets_epsilon_sample_r_bis = np.average([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_bis)])
        sample_r_model_meets_epsilon_acc_bis = np.average([results_all_val[modelS][fold]["train_acc"] for fold, modelS in enumerate(folds_selection_bis)])
        if logPlot:
            good_index_bis = find_position_log(steps, overall_sample_robustness_val, sample_r_model_meets_epsilon_sample_r_bis, -1)
        else:
            good_index_bis = find_position(steps, overall_sample_robustness, sample_r_model_meets_epsilon_sample_r_bis, -1) # last parameter not used anymore
        ax.plot(good_index_bis, sample_r_model_meets_epsilon_sample_r_bis, marker='o', color=color_val_train, label='val (train unf criteria)')
        ax2.plot(good_index_bis, sample_r_model_meets_epsilon_acc_bis, marker=train_acc_marker, color=color_val_train, label='val (train unf criteria)')

        # before constant classifier criteria
        before_constant_sample_r = np.average([results_all[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_ter)])
        before_constant_acc = np.average([results_all[modelS][fold]["train_acc"] for fold, modelS in enumerate(folds_selection_ter)])
        if logPlot:
            good_index_ter = find_position_log(steps, overall_sample_robustness, before_constant_sample_r, -1)
        else:
            good_index_ter = find_position(steps, overall_sample_robustness, before_constant_sample_r, -1) # last parameter not used anymore
        ax.plot(good_index_ter, before_constant_sample_r, marker='o', color=color_noval_trivial, label='noval (before-constant criteria)')
        ax2.plot(good_index_ter, before_constant_acc, marker=train_acc_marker, color=color_noval_trivial, label='noval (before-constant criteria)')

        # Uses the same position for unfairness
        #plt.plot(good_index_ter, before_constant_unf_train, marker=train_unf_marker, color=color_noval_trivial)
        #plt.plot(good_index_ter, before_constant_unf, marker=test_unf_marker, color=color_noval_trivial)

    if plotType == 840:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legendFig = plt.figure("Legend plot")
        #ax = fig.add_subplot(111)
        legend_elements = [
                    Line2D([0], [0], color=color_noval_all, lw=4, label='sample robust fair frontier (no validation)'),
                    Line2D([0], [0], color=color_noval_trivial, lw=4, label='no validation (before-constant)'),
                    Line2D([0], [0], color=color_val_all, lw=4, label='sample robust fair frontier (validation)'),
                    Line2D([0], [0], color=color_val_eps, lw=4,  label='validation ($\epsilon$ criterion)'),
                    Line2D([0], [0], color=color_val_train, lw=4, label='validation (train unf. criterion)'),
                    Line2D([0], [0], marker='o', color='black', lw=lw_general, label='train sample robustness'),
                    Line2D([0], [0], marker=train_acc_marker, color='black', lw=lw_general, label='train accuracy'),
                    Line2D([0], [0], marker=test_acc_marker, color='black', lw=lw_test,  linestyle = '--', label='test accuracy')]     
                    #Line2D([0], [0], color=colors_masks[0], lw=4, label='no mask'),
                    #Line2D([0], [0], color=colors_masks[10], lw=4, label='10 masks'),
                    #Line2D([0], [0], color=colors_masks[30], lw=4, label='30 masks'),

        '''
        color_noval_all = 'grey'
        color_noval_trivial  = 'red'
        color_val_eps = 'skyblue'
        color_val_train = 'darkblue'
        color_val_all = 'darkviolet'
        color_val_all_val = 'magenta'
        colors_masks = {}
        colors_masks[0] = 'gold' #'green'
        colors_masks[10]= 'lime' #'magenta'
        colors_masks[30] = 'darkgreen' #'gold'
        color_epsilon = 'orange'
        train_unf_marker = 'D'
        test_unf_marker = 'o'
        '''                                                        

        legendFig.legend(handles=legend_elements, loc='center', ncol=3)
        if showArg>0:
            legendFig.show()
        else:
            legendFig.savefig('./Paretos_All_Figures_Minor_Revision/sample-robustness-train_accuracy_steps_legend_revised.pdf', bbox_inches='tight')
        exit()
        '''else:
        handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]

        handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
        fig.legend(handles, labels, loc="lower center",   # Position of legend
            borderaxespad=0.1, ncol=3) #ncol=2)'''
    #fig.subplots_adjust(bottom=0.2)
    #plt.title("Dataset %s, metric %s, $\epsilon=%f$" %(datasetNames[dataset], fairnessMetricName[fairnessMetric-1], epsilon))
    if title=="metric":
        plt.title("%s metric" %(fairnessMetricName[fairnessMetric-1]))
    elif title=="dataset":
        plt.title("%s dataset" %(datasetNames[dataset]))
    if showArg>0:
        plt.show()
    else:
        plt.savefig("./Paretos_All_Figures_Minor_Revision/sample-robustness-steps_train_acc_dataset_%s-metric_%d-epsilon_%f_revised.%s" %(dataset, fairnessMetric, epsilon, extension), bbox_inches='tight')

# Train sample-robustness and train accuracy, minor revision -> INCLUDE VALIDATION OR TEST DATA ACCURACY + STD for test
elif plotType == 420 or plotType == 840:
    fig,ax = plt.subplots()
    overall_sample_robustness = [results_all[i][5]["train_sample_robustness"] for i in range(step)]
    ax.plot(steps, overall_sample_robustness, marker='o', markersize=markersize_value, color=color_noval_all, label="train fairness sample robustness")
    #ax.legend(loc='best')
    ax.set_ylabel("Train Fairness Sample Robustness (log)")
    plt.xlabel("Steps for the Fairness Sample-Robust Frontier")
    logPlot = True
    if logPlot:
        ax.set_yscale("log")

    ax2=ax.twinx()
    plt.ylabel("Accuracy (test and train)")
    overall_err_train = [results_all[i][5]["train_acc"] for i in range(step)]
    overall_err_test = [results_all[i][5]["test_acc"] for i in range(step)]
    ax2.plot(steps, overall_err_train, color=color_noval_all, marker=train_acc_marker, markersize=markersize_value, label="train unfairness", linewidth=lw_general)#, marker=train_unf_marker)
    # Below: ajout minor revision
    ax2.plot(steps, overall_err_test, '--', color=color_noval_all, marker=test_acc_marker, markersize=markersize_value, label="train unfairness", linewidth=lw_test)#, marker=train_unf_marker)
    if False:
        for m in masks:
            # Find good position for sample-robustness
            if logPlot:
                good_index = find_position_log(steps, overall_sample_robustness, results_masks[m][5]["train_sample_robustness"], m)
            else:
                good_index = find_position(steps, overall_sample_robustness, results_masks[m][5]["train_sample_robustness"], m)
            ax.plot(good_index, results_masks[m][5]["train_sample_robustness"], color=colors_masks[m], marker='o', label='%d masks' %m)
            # Uses the same position for unfairness
            #ax2.scatter(good_index, results_masks[m][5]["train_sample_robustness"], color=colors_masks[m], marker=train_unf_marker)
            #ax2.scatter(good_index, results_masks[m][5]["train_acc"], color=colors_masks[m], marker=test_unf_marker)

    if includeValidation:
        # all data curves
        steps_val = [i for i in range(len(results_all_val))]
        overall_sample_robustness_val = [results_all_val[i][5]["train_sample_robustness"] for i in steps_val]
        ax.plot(steps_val, overall_sample_robustness_val, marker='o', markersize=markersize_value, color=color_val_all, label="validation", linewidth=lw_general)#, marker=train_unf_marker)
        ax2.plot(steps_val, [results_all_val[i][5]["train_acc"] for i in steps_val], marker=train_acc_marker, markersize=markersize_value, color=color_val_all, label="validation", linewidth=lw_general)#, marker=train_unf_marker)
        #plt.plot(steps_val, [results_all_val[i][5]["val_unf"] for i in steps_val], ':', color=color_val_all_val, label="validation unfairness", linewidth=lw_general)#, marker=train_unf_marker)
        # Below: ajout minor revision
        ax2.plot(steps_val, [results_all_val[i][5]["test_acc"] for i in steps_val], '--', marker=test_acc_marker, markersize=markersize_value, color=color_val_all, label="validation", linewidth=lw_test)#, marker=train_unf_marker)

        # epsilon criteria
        sample_r_model_meets_epsilon_sample_r = np.average([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection)])
        sample_r_model_meets_epsilon_acc = np.average([results_all_val[modelS][fold]["train_acc"] for fold, modelS in enumerate(folds_selection)])
        if logPlot:
            good_index = find_position_log(steps, overall_sample_robustness_val, sample_r_model_meets_epsilon_sample_r, -1)
        else:
            good_index = find_position(steps, overall_sample_robustness, sample_r_model_meets_epsilon_sample_r, -1) # last parameter not used anymore
        ax.plot(good_index, sample_r_model_meets_epsilon_sample_r, marker='o', color=color_val_eps, label='val (epsilon criteria)')
        ax2.plot(good_index, sample_r_model_meets_epsilon_acc, marker=train_acc_marker, color=color_val_eps, label='val (train unf criteria)')
        ax2.errorbar(x=good_index, y =original_unf, yerr=original_unf_std, color=color_original)#, lw=1)

        # train fairness criteria
        sample_r_model_meets_epsilon_sample_r_bis = np.average([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_bis)])
        sample_r_model_meets_epsilon_acc_bis = np.average([results_all_val[modelS][fold]["train_acc"] for fold, modelS in enumerate(folds_selection_bis)])
        if logPlot:
            good_index_bis = find_position_log(steps, overall_sample_robustness_val, sample_r_model_meets_epsilon_sample_r_bis, -1)
        else:
            good_index_bis = find_position(steps, overall_sample_robustness, sample_r_model_meets_epsilon_sample_r_bis, -1) # last parameter not used anymore
        ax.plot(good_index_bis, sample_r_model_meets_epsilon_sample_r_bis, marker='o', color=color_val_train, label='val (train unf criteria)')
        ax2.plot(good_index_bis, sample_r_model_meets_epsilon_acc_bis, marker=train_acc_marker, color=color_val_train, label='val (train unf criteria)')

        # before constant classifier criteria
        before_constant_sample_r = np.average([results_all[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_ter)])
        before_constant_acc = np.average([results_all[modelS][fold]["train_acc"] for fold, modelS in enumerate(folds_selection_ter)])
        if logPlot:
            good_index_ter = find_position_log(steps, overall_sample_robustness, before_constant_sample_r, -1)
        else:
            good_index_ter = find_position(steps, overall_sample_robustness, before_constant_sample_r, -1) # last parameter not used anymore
        ax.plot(good_index_ter, before_constant_sample_r, marker='o', color=color_noval_trivial, label='noval (before-constant criteria)')
        ax2.plot(good_index_ter, before_constant_acc, marker=train_acc_marker, color=color_noval_trivial, label='noval (before-constant criteria)')

        # Uses the same position for unfairness
        #plt.plot(good_index_ter, before_constant_unf_train, marker=train_unf_marker, color=color_noval_trivial)
        #plt.plot(good_index_ter, before_constant_unf, marker=test_unf_marker, color=color_noval_trivial)

    if plotType == 840:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legendFig = plt.figure("Legend plot")
        #ax = fig.add_subplot(111)
        legend_elements = [
                    Line2D([0], [0], color=color_noval_trivial, lw=4, label='no validation (before-constant)'),
                    Line2D([0], [0], color=color_val_eps, lw=4,  label='validation ($\epsilon$ criterion)'),
                    Line2D([0], [0], color=color_val_train, lw=4, label='validation (train unf. criterion)'),
                    Line2D([0], [0], color=color_noval_all, lw=4, label='sample robust fair frontier (no validation)'),
                    Line2D([0], [0], color=color_val_all, lw=4, label='sample robust fair frontier (validation)'),
                    Line2D([0], [0], marker='o', color='black', lw=lw_general, label='train sample robustness'),
                    Line2D([0], [0], marker=train_acc_marker, color='black', lw=lw_general, label='train accuracy'),
                    Line2D([0], [0], marker=test_acc_marker, color='black', lw=lw_test,  linestyle = '--', label='test accuracy')]     
                    #Line2D([0], [0], color=colors_masks[0], lw=4, label='no mask'),
                    #Line2D([0], [0], color=colors_masks[10], lw=4, label='10 masks'),
                    #Line2D([0], [0], color=colors_masks[30], lw=4, label='30 masks'),

        '''
        color_noval_all = 'grey'
        color_noval_trivial  = 'red'
        color_val_eps = 'skyblue'
        color_val_train = 'darkblue'
        color_val_all = 'darkviolet'
        color_val_all_val = 'magenta'
        colors_masks = {}
        colors_masks[0] = 'gold' #'green'
        colors_masks[10]= 'lime' #'magenta'
        colors_masks[30] = 'darkgreen' #'gold'
        color_epsilon = 'orange'
        train_unf_marker = 'D'
        test_unf_marker = 'o'
        '''                                                        

        legendFig.legend(handles=legend_elements, loc='center', ncol=3)
        if showArg>0:
            legendFig.show()
        else:
            legendFig.savefig('./Paretos_All_Figures_Minor_Revision/sample-robustness-train_accuracy_steps_legend_revised.pdf', bbox_inches='tight')
        exit()
        '''else:
        handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]

        handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
        fig.legend(handles, labels, loc="lower center",   # Position of legend
            borderaxespad=0.1, ncol=3) #ncol=2)'''
    #fig.subplots_adjust(bottom=0.2)
    #plt.title("Dataset %s, metric %s, $\epsilon=%f$" %(datasetNames[dataset], fairnessMetricName[fairnessMetric-1], epsilon))
    if title=="metric":
        plt.title("%s metric" %(fairnessMetricName[fairnessMetric-1]))
    elif title=="dataset":
        plt.title("%s dataset" %(datasetNames[dataset]))
    if showArg>0:
        plt.show()
    else:
        plt.savefig("./Paretos_All_Figures_Minor_Revision/sample-robustness-steps_train_acc_dataset_%s-metric_%d-epsilon_%f_revised.%s" %(dataset, fairnessMetric, epsilon, extension), bbox_inches='tight')

## Overall fairness sample-robustness + overall fairness train AND test violation, minor revision -> INCLUDE STD ?
if plotType == 600 or plotType == 1200:

    #fig,ax = plt.subplots()
    overall_sample_robustness = [results_all[i][5]["train_sample_robustness"] for i in range(step)]
    #ax.plot(steps, overall_sample_robustness, '--', marker='o', color=color_noval_all, label="train fairness sample robustness")
    #ax.legend(loc='best')
    #ax.set_ylabel("Train Fairness Sample Robustness (log)")
    plt.xlabel("Steps for the Fairness Sample-Robust Frontier")
    logPlot = False
    if logPlot:
        plt.yscale("log")

    #ax2=ax.twinx()
    plt.ylabel("Unfairness")
    overall_unf_train = [results_all[i][5]["train_unf"] for i in range(step)]
    plt.plot(steps, overall_unf_train, color=color_noval_all, label="train unfairness", linewidth=lw_general)#, marker=train_unf_marker)
    overall_unf_test = [results_all[i][5]["test_unf"] for i in range(step)]
    plt.plot(steps, overall_unf_test, '--', color=color_noval_all, label="test unfairness", linewidth=lw_general)#, marker='o')
    #overall_unf_val = [results_all_val[i][5]["val_unf"] for i in range(len(results_all_val))]
    #ax2.plot([k for k in range(len(results_all_val))], overall_unf_val, marker='o', color='olive', label="validation unfairness")
    plt.plot([i for i in range(max([step, len(results_all_val)]))], [1.0-epsilon for i in range(max([step, len(results_all_val)]))], color=color_epsilon, label="epsilon", linewidth=lw_general)
    #ax2.legend(loc='best')
    if False:
        for m in masks:
            # Find good position for sample-robustness
            if logPlot:
                good_index = find_position_log(steps, overall_sample_robustness, results_masks[m][5]["train_sample_robustness"], m)
            else:
                good_index = find_position(steps, overall_sample_robustness, results_masks[m][5]["train_sample_robustness"], m)
            #ax.plot(good_index, results_masks[m][5]["train_sample_robustness"], color=colors_masks[m], marker='o', label='%d masks' %m)
            # Uses the same position for unfairness
            plt.plot(good_index, results_masks[m][5]["train_unf"], color=colors_masks[m], marker=train_unf_marker)
            plt.plot(good_index, results_masks[m][5]["test_unf"], color=colors_masks[m], marker=test_unf_marker)

    # Original FairCORELS
    original_sample_r = np.average([results_all[0][fold]["train_sample_robustness"] for fold in range(5)])
    original_unf = np.average([results_all[0][fold]["test_unf"] for fold in range(5)])
    original_unf_std = np.std([results_all[0][fold]["test_unf"] for fold in range(5)])
    print("original_unf=",  original_unf, "+-", original_unf_std)
    original_train_unf = np.average([results_all[0][fold]["train_unf"] for fold in range(5)])
    original_train_unf_std = np.std([results_all[0][fold]["train_unf"] for fold in range(5)])
    print("original_train_unf=",  original_train_unf, "+-", original_train_unf_std)
    if logPlot:
        good_index_original = find_position_log(steps, overall_sample_robustness, sample_r_model_meets_epsilon_sample_r, -1)
    else:
        good_index_original = 0
    # Uses the same position for unfairness
    color_original = "yellow"
    plt.plot(good_index_original, original_train_unf, marker=train_unf_marker, color=color_original)
    #plt.errorbar(x=good_index_original+0.5, y =original_train_unf, yerr=original_train_unf_std, color=color_original)#, lw=1)
    plt.plot(good_index_original, original_unf, marker=test_unf_marker, color=color_original)
    plt.errorbar(x=good_index_original, y =original_unf, yerr=original_unf_std, color=color_original)#, lw=1)

    if includeValidation:
        # all data curves
        steps_val = [i for i in range(len(results_all_val))]
        plt.plot(steps_val, [results_all_val[i][5]["train_unf"] for i in steps_val], color=color_val_all, label="train unfairness", linewidth=lw_general)#, marker=train_unf_marker)
        plt.plot(steps_val, [results_all_val[i][5]["test_unf"] for i in steps_val], '--', color=color_val_all, label="test unfairness", linewidth=lw_general)#, marker=train_unf_marker)
        plt.plot(steps_val, [results_all_val[i][5]["val_unf"] for i in steps_val], ':', color=color_val_all_val, label="validation unfairness", linewidth=lw_general)#, marker=train_unf_marker)

        # epsilon criteria
        sample_r_model_meets_epsilon_sample_r = np.average([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection)])
        sample_r_model_meets_epsilon_unf = np.average([results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection)])
        sample_r_model_meets_epsilon_unf_std = np.std([results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection)])
        print("sample_r_model_meets_epsilon_unf=",  sample_r_model_meets_epsilon_unf, "+-", sample_r_model_meets_epsilon_unf_std)
        sample_r_model_meets_epsilon_unf_train = np.average([results_all_val[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection)])
        sample_r_model_meets_epsilon_unf_train_std = np.std([results_all_val[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection)])
        print("sample_r_model_meets_epsilon_unf_train=",  sample_r_model_meets_epsilon_unf_train, "+-", sample_r_model_meets_epsilon_unf_train_std)
        if logPlot:
            good_index = find_position_log(steps, overall_sample_robustness, sample_r_model_meets_epsilon_sample_r, -1)
        else:
            #good_index = find_position(steps, overall_sample_robustness, sample_r_model_meets_epsilon_sample_r, -1) # last parameter not used anymore
            good_index = np.average(folds_selection)
        #ax.plot(good_index, sample_r_model_meets_epsilon_sample_r, marker='o', color=color_val_eps, label='val (epsilon criteria)')
        # Uses the same position for unfairness
        plt.plot(good_index, sample_r_model_meets_epsilon_unf_train, marker=train_unf_marker, color=color_val_eps)
        #plt.errorbar(x=good_index+0.5, y =sample_r_model_meets_epsilon_unf_train, yerr=sample_r_model_meets_epsilon_unf_train_std, color=color_val_eps)#, lw=1)
        
        plt.plot(good_index, sample_r_model_meets_epsilon_unf, marker=test_unf_marker, color=color_val_eps)
        plt.errorbar(x=good_index, y =sample_r_model_meets_epsilon_unf, yerr=sample_r_model_meets_epsilon_unf_std, color=color_val_eps)#, lw=1)

        # train fairness criteria
        sample_r_model_meets_epsilon_sample_r_bis = np.average([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_bis)])
        sample_r_model_meets_epsilon_unf_bis = np.average([results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_bis)])
        sample_r_model_meets_epsilon_unf_bis_std = np.std([results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_bis)])
        print("sample_r_model_meets_epsilon_unf_bis=",  sample_r_model_meets_epsilon_unf_bis, "+-", sample_r_model_meets_epsilon_unf_bis_std)
        sample_r_model_meets_epsilon_unf_train_bis = np.average([results_all_val[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection_bis)])
        sample_r_model_meets_epsilon_unf_train_bis_std = np.std([results_all_val[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection_bis)])
        print("sample_r_model_meets_epsilon_unf_train_bis=",  sample_r_model_meets_epsilon_unf_train_bis, "+-", sample_r_model_meets_epsilon_unf_train_bis_std)
        if logPlot:
            good_index_bis = find_position_log(steps, overall_sample_robustness, sample_r_model_meets_epsilon_sample_r_bis, -1)
        else:
            #good_index_bis = find_position(steps, overall_sample_robustness, sample_r_model_meets_epsilon_sample_r_bis, -1) # last parameter not used anymore
            good_index_bis = np.average(folds_selection_bis)
        #ax.plot(good_index_bis, sample_r_model_meets_epsilon_sample_r_bis, marker='o', color=color_val_train, label='val (train unf criteria)')
        
        plt.plot(good_index_bis, sample_r_model_meets_epsilon_unf_train_bis, marker=train_unf_marker, color=color_val_train)
        #plt.errorbar(x=good_index_bis+0.5, y =sample_r_model_meets_epsilon_unf_train_bis, yerr=sample_r_model_meets_epsilon_unf_train_bis_std, color=color_val_train)#, lw=1)

        plt.plot(good_index_bis, sample_r_model_meets_epsilon_unf_bis, marker=test_unf_marker, color=color_val_train)
        plt.errorbar(x=good_index_bis, y =sample_r_model_meets_epsilon_unf_bis, yerr=sample_r_model_meets_epsilon_unf_bis_std, color=color_val_train)#, lw=1)

        # before constant classifier criteria
        before_constant_sample_r = np.average([results_all[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_ter)])
        before_constant_unf = np.average([results_all[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_ter)])
        before_constant_unf_std = np.std([results_all[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_ter)])
        print("before_constant_unf=",  before_constant_unf, "+-", before_constant_unf_std)
        before_constant_unf_train = np.average([results_all[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection_ter)])
        before_constant_unf_train_std = np.std([results_all[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection_ter)])
        print("before_constant_unf_train=", before_constant_unf_train, "+-",  before_constant_unf_train_std)
        if logPlot:
            good_index_ter = find_position_log(steps, overall_sample_robustness, before_constant_sample_r, -1)
        else:
            #good_index_ter = find_position(steps, overall_sample_robustness, before_constant_sample_r, -1) # last parameter not used anymore
            good_index_ter = np.average(folds_selection_ter)
        #ax.plot(good_index_ter, before_constant_sample_r, marker='o', color='black', label='noval (before-constant criteria)')

        # Uses the same position for unfairness
        plt.plot(good_index_ter, before_constant_unf_train, marker=train_unf_marker, color=color_noval_trivial)
        #plt.errorbar(x=good_index_ter+0.5, y =before_constant_unf_train, yerr=before_constant_unf_train_std, color=color_noval_trivial)#, lw=1)

        plt.plot(good_index_ter, before_constant_unf, marker=test_unf_marker, color=color_noval_trivial)
        plt.errorbar(x=good_index_ter, y =before_constant_unf, yerr=before_constant_unf_std, color=color_noval_trivial)#, lw=1)

    if plotType == 12:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legendFig = plt.figure("Legend plot")
        #ax = fig.add_subplot(111)
        legend_elements = [
                    Line2D([0], [0], color=color_noval_trivial, lw=4, label='no validation (before-constant)'),
                    Line2D([0], [0], color=color_val_eps, lw=4,  label='validation ($\epsilon$ criterion)'),
                    Line2D([0], [0], color=color_val_train, lw=4, label='validation (train unf. criterion)'),
                    Line2D([0], [0], color=color_noval_all, lw=4, label='sample robust fair frontier (no validation)'),
                    Line2D([0], [0], color=color_val_all, lw=4, label='sample robust fair frontier (validation)'),
                    Line2D([0], [0], color=color_epsilon, lw=1, label='$\epsilon$'),
                    Line2D([0], [0], marker=None, color=color_val_all_val, lw=1, linestyle = ':', label='validation unfairness'),
                    Line2D([0], [0], marker=train_unf_marker, color='black', lw=1, label='train unfairness'),
                    Line2D([0], [0], marker=test_unf_marker, color='black', lw=1,  linestyle = '--', label='test unfairness')]     
                    #Line2D([0], [0], color=colors_masks[0], lw=4, label='no mask'),
                    #Line2D([0], [0], color=colors_masks[10], lw=4, label='10 masks'),
                    #Line2D([0], [0], color=colors_masks[30], lw=4, label='30 masks'),

        '''
        color_noval_all = 'grey'
        color_noval_trivial  = 'red'
        color_val_eps = 'skyblue'
        color_val_train = 'darkblue'
        color_val_all = 'darkviolet'
        color_val_all_val = 'magenta'
        colors_masks = {}
        colors_masks[0] = 'gold' #'green'
        colors_masks[10]= 'lime' #'magenta'
        colors_masks[30] = 'darkgreen' #'gold'
        color_epsilon = 'orange'
        train_unf_marker = 'D'
        test_unf_marker = 'o'
        '''                                                        

        legendFig.legend(handles=legend_elements, loc='center', ncol=3)
        if showArg>0:
            legendFig.show()
        else:
            legendFig.savefig('./Paretos_All_Figures/sample-robustness-steps_legend.pdf', bbox_inches='tight')
        exit()
        '''else:
        handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]

        handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
        fig.legend(handles, labels, loc="lower center",   # Position of legend
            borderaxespad=0.1, ncol=3) #ncol=2)'''
    #fig.subplots_adjust(bottom=0.2)
    #plt.title("Dataset %s, metric %s, $\epsilon=%f$" %(datasetNames[dataset], fairnessMetricName[fairnessMetric-1], epsilon))
    if title=="metric":
        plt.title("%s metric" %(fairnessMetricName[fairnessMetric-1]))
    elif title=="dataset":
        plt.title("%s dataset" %(datasetNames[dataset]))
    if showArg>0:
        plt.show()
    else:
        plt.savefig("./Paretos_All_Figures_Minor_Revision/sample-robustness-steps_dataset_%s-metric_%d-epsilon_%f.%s" %(dataset, fairnessMetric, epsilon, extension), bbox_inches='tight')
if plotType == 601 or plotType == 1201:
    # Original FairCORELS
    original_sample_r = np.average([results_all[0][fold]["train_sample_robustness"] for fold in range(5)])
    original_sample_r_std = np.std([results_all[0][fold]["train_sample_robustness"] for fold in range(5)])
    print("original_sample_r=",original_sample_r,"+-",original_sample_r_std)

    original_test_unf = np.average([1.0-results_all[0][fold]["test_unf"] for fold in range(5)])
    original_test_unf_std = np.std([1.0-results_all[0][fold]["test_unf"] for fold in range(5)])
    print("original_test_unf=",  original_test_unf, "+-", original_test_unf_std)
    
    original_train_unf = np.average([1.0-results_all[0][fold]["train_unf"] for fold in range(5)])
    original_train_unf_std = np.std([1.0-results_all[0][fold]["train_unf"] for fold in range(5)]) 
    print("original_train_unf=",  original_train_unf, "+-", original_train_unf_std)

    original_test_acc = np.average([1.0-results_all[0][fold]["test_acc"] for fold in range(5)])
    original_test_acc_std = np.std([1.0-results_all[0][fold]["test_acc"] for fold in range(5)])
    print("original_test_acc=",  original_test_acc, "+-", original_test_acc_std)

    original_train_acc = np.average([results_all[0][fold]["train_acc"] for fold in range(5)])
    original_train_acc_std = np.std([results_all[0][fold]["train_acc"] for fold in range(5)]) 
    print("original_train_acc=",  original_train_acc, "+-", original_train_acc_std)
   
    # epsilon criteria
    validation_epsilon_criterion_sample_r = np.average([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection)])
    validation_epsilon_criterion_sample_r_std = np.std([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection)])
    print("validation_epsilon_criterion_sample_r=",validation_epsilon_criterion_sample_r,"+-",validation_epsilon_criterion_sample_r_std)

    validation_epsilon_criterion_test_unf = np.average([1.0-results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection)])
    validation_epsilon_criterion_test_unf_std = np.std([1.0-results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection)])
    print("validation_epsilon_criterion_test_unf=",  validation_epsilon_criterion_test_unf, "+-", validation_epsilon_criterion_test_unf_std)

    validation_epsilon_criterion_train_unf = np.average([1.0-results_all_val[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection)])
    validation_epsilon_criterion_train_unf_std = np.std([1.0-results_all_val[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection)])
    print("validation_epsilon_criterion_train_unf=",  validation_epsilon_criterion_train_unf, "+-", validation_epsilon_criterion_train_unf_std)

    validation_epsilon_criterion_test_acc = np.average([results_all_val[modelS][fold]["test_acc"] for fold, modelS in enumerate(folds_selection)])
    validation_epsilon_criterion_test_acc_std = np.std([results_all_val[modelS][fold]["test_acc"] for fold, modelS in enumerate(folds_selection)])
    print("validation_epsilon_criterion_test_acc=",  validation_epsilon_criterion_test_acc, "+-", validation_epsilon_criterion_test_acc_std)

    validation_epsilon_criterion_train_acc = np.average([results_all_val[modelS][fold]["train_acc"] for fold, modelS in enumerate(folds_selection)])
    validation_epsilon_criterion_train_acc_std = np.std([results_all_val[modelS][fold]["train_acc"] for fold, modelS in enumerate(folds_selection)])
    print("validation_epsilon_criterion_train_acc=",  validation_epsilon_criterion_train_acc, "+-", validation_epsilon_criterion_train_acc_std)


    # train fairness criteria
    validation_trainunf_criterion_sample_r = np.average([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_bis)])
    validation_trainunf_criterion_sample_r_std = np.std([results_all_val[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_bis)])
    print("validation_trainunf_criterion_sample_r=",validation_trainunf_criterion_sample_r,"+-",validation_trainunf_criterion_sample_r_std)

    validation_trainunf_criterion_test_unf = np.average([1.0-results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_bis)])
    validation_trainunf_criterion_test_unf_std = np.std([1.0-results_all_val[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_bis)])
    print("validation_trainunf_criterion_test_unf=",  validation_trainunf_criterion_test_unf, "+-", validation_trainunf_criterion_test_unf_std)

    validation_trainunf_criterion_train_unf = np.average([1.0-results_all_val[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection_bis)])
    validation_trainunf_criterion_train_unf_std = np.std([1.0-results_all_val[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection_bis)])
    print("validation_trainunf_criterion_train_unf=",  validation_trainunf_criterion_train_unf, "+-", validation_trainunf_criterion_train_unf_std)

    validation_trainunf_criterion_test_acc = np.average([results_all_val[modelS][fold]["test_acc"] for fold, modelS in enumerate(folds_selection_bis)])
    validation_trainunf_criterion_test_acc_std = np.std([results_all_val[modelS][fold]["test_acc"] for fold, modelS in enumerate(folds_selection_bis)])
    print("validation_trainunf_criterion_test_acc=",  validation_trainunf_criterion_test_acc, "+-", validation_trainunf_criterion_test_acc_std)

    validation_trainunf_criterion_train_acc = np.average([results_all_val[modelS][fold]["train_acc"] for fold, modelS in enumerate(folds_selection_bis)])
    validation_trainunf_criterion_train_acc_std = np.std([results_all_val[modelS][fold]["train_acc"] for fold, modelS in enumerate(folds_selection_bis)])
    print("validation_trainunf_criterion_train_acc=",  validation_trainunf_criterion_train_acc, "+-", validation_trainunf_criterion_train_acc_std)

    # before constant classifier criteria
    before_constant_sample_r = np.average([results_all[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_ter)])
    before_constant_sample_r_std = np.std([results_all[modelS][fold]["train_sample_robustness"] for fold, modelS in enumerate(folds_selection_ter)])
    print("before_constant_sample_r=",before_constant_sample_r,"+-",before_constant_sample_r_std)

    before_constant_test_unf = np.average([results_all[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_ter)])
    before_constant_test_unf_std = np.std([results_all[modelS][fold]["test_unf"] for fold, modelS in enumerate(folds_selection_ter)])
    print("before_constant_test_unf=",  before_constant_test_unf, "+-", before_constant_test_unf_std)

    before_constant_train_unf = np.average([1.0-results_all[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection_ter)])
    before_constant_train_unf_std = np.std([1.0-results_all[modelS][fold]["train_unf"] for fold, modelS in enumerate(folds_selection_ter)])
    print("before_constant_train_unf=", before_constant_train_unf, "+-",  before_constant_train_unf_std)

    before_constant_test_acc = np.average([1.0-results_all[modelS][fold]["test_acc"] for fold, modelS in enumerate(folds_selection_ter)])
    before_constant_test_acc_std = np.std([1.0-results_all[modelS][fold]["test_acc"] for fold, modelS in enumerate(folds_selection_ter)])
    print("before_constant_test_acc=",  before_constant_test_acc, "+-", before_constant_test_acc_std)

    before_constant_train_acc = np.average([1.0-results_all[modelS][fold]["train_acc"] for fold, modelS in enumerate(folds_selection_ter)])
    before_constant_train_acc_std = np.std([1.0-results_all[modelS][fold]["train_acc"] for fold, modelS in enumerate(folds_selection_ter)])
    print("before_constant_train_acc=", before_constant_train_acc, "+-",  before_constant_train_acc_std)

    with open('./Paretos_All_Figures_Minor_Revision/results_with_std-metric_%d-epsilon_%f-dataset_%s.csv' %(args.metric, args.epsilon, args.dataset), mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Method", "Train SR.", "Train Acc.", "Test Acc.", "Train Unf.", "Test Unf."])
        csv_writer.writerow(["Original", "$%.4f \pm %.4f$" %(original_sample_r, original_sample_r_std), 
        "$%.4f \pm %.4f$" %(original_train_acc, original_train_acc_std), 
        "$%.4f \pm %.4f$" %(original_test_acc, original_test_acc_std),
        "$%.4f \pm %.4f$" %(original_train_unf, original_train_unf_std), 
        "$%.4f \pm %.4f$" %(original_test_unf, original_test_unf_std)])

        csv_writer.writerow(["val (eps)", "$%.4f \pm %.4f$" %(validation_epsilon_criterion_sample_r, validation_epsilon_criterion_sample_r_std), 
        "$%.4f \pm %.4f$" %(validation_epsilon_criterion_train_acc, validation_epsilon_criterion_train_acc_std), 
        "$%.4f \pm %.4f$" %(validation_epsilon_criterion_test_acc, validation_epsilon_criterion_test_acc_std),
        "$%.4f \pm %.4f$" %(validation_epsilon_criterion_train_unf, validation_epsilon_criterion_train_unf_std), 
        "$%.4f \pm %.4f$" %(validation_epsilon_criterion_test_unf, validation_epsilon_criterion_test_unf_std)])

        csv_writer.writerow(["val (train unf)", "$%.4f \pm %.4f$" %(validation_trainunf_criterion_sample_r, validation_trainunf_criterion_sample_r_std), 
        "$%.4f \pm %.4f$" %(validation_trainunf_criterion_train_acc, validation_trainunf_criterion_train_acc_std), 
        "$%.4f \pm %.4f$" %(validation_trainunf_criterion_test_acc, validation_trainunf_criterion_test_acc_std),
        "$%.4f \pm %.4f$" %(validation_trainunf_criterion_train_unf, validation_trainunf_criterion_train_unf_std), 
        "$%.4f \pm %.4f$" %(validation_trainunf_criterion_test_unf, validation_trainunf_criterion_test_unf_std)])

        csv_writer.writerow(["no val before constant", "$%.4f \pm %.4f$" %(before_constant_sample_r, before_constant_sample_r_std), 
        "$%.4f \pm %.4f$" %(before_constant_train_acc, before_constant_train_acc_std), 
        "$%.4f \pm %.4f$" %(before_constant_test_acc, before_constant_test_acc_std),
        "$%.4f \pm %.4f$" %(before_constant_train_unf, before_constant_train_unf_std), 
        "$%.4f \pm %.4f$" %(before_constant_test_unf, before_constant_test_unf_std)])
else:
    print(plotType)
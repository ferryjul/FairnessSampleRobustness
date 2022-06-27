
datasets=["adult", "compas", "default_credit", "marketing"]#]
heuristic = "BFS-objective-aware"
solvers = ['Mistral', 'OR-tools']
fairnessMetricName=["Statistical Parity", "Predictive Parity", "Predictive Equality", "Equal Opportunity", "Equalized Odds"]
solver_arg = solvers[1]

readMesures = 0
readSubOptimalHeuristic = 0
import pandas as pd

gaps = []
readSubOptimalMIP = 0
        
for fairnessMetric in [1, 3, 4, 5]:
    for dataset in datasets:
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

        
        for i in range(len(fileContent.values)):
            for j in [1,2,3]:
                if fileContent.values[i][j] == fileContentHeuristic.values[i][j]:
                    readMesures+=1
                elif fileContent.values[i][j] < fileContentHeuristic.values[i][j]:
                    readMesures+=1
                    readSubOptimalHeuristic+=1
                    gap = (fileContentHeuristic.values[i][j]-fileContent.values[i][j])/fileContent.values[i][j]
                    gaps.append(gap)
                elif fileContent.values[i][j] > fileContentHeuristic.values[i][j]:
                    readMesures+=1
                    readSubOptimalMIP+=1
print("Heuristic found suboptimal result %d/%d = %d percent of audits" %(readSubOptimalHeuristic, readMesures, 100*readSubOptimalHeuristic/readMesures)  )
if readSubOptimalHeuristic>0:
    print("(Average gap is %f)"%(sum(gaps)/len(gaps)))
print("MIP found suboptimal result %d/%d = %d percent of audits" %(readSubOptimalMIP, readMesures, 100*readSubOptimalMIP/readMesures)  )
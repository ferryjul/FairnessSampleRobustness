import pandas as pd 
import csv
import numpy as np
import argparse
# parser initialization
parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')

parser.add_argument('--expe', type=int, default=-1, help='in {1, 2, 3}')

args = parser.parse_args()

expe = args.expe
#expe = 3
nb_seed = 100
n_modes = 6
algos = ["_algo3", "_algo4"]#", "_algo4"]
n_digits_violation = 3
seed = 0
nb_read = 0
nb_models = 1
if expe == 1:
    metric= "ppr" #'eo'
    folder= "runAdultPPR" #"results-new"
    dataset = "adult"
    modelType = "dnn50"#"linear"
    epsList = [0.8]#, 0.0001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
elif expe == 2:
    metric= "tpr" #'eo'
    folder= "runCompasTPR" #"results-new"
    dataset = "compas"
    modelType = "dnn50"#"linear"
    epsList = [1.05]#, 0.0001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
elif expe == 3:
    metric= "fpr" #'eo'
    folder= "runs_wave2_marketing_200epochs_fpr" #"results-new"
    dataset = "marketing_200epochs"
    modelType = "dnn"#"linear"
    epsList = [0.0]#, 0.0001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    n_digits_violation = 4
elif expe == 4:
    metric= "min_tpr" #'eo'
    folder= "default_credit_min_tpr_0.5" #"results-new"
    dataset = "default_credit"
    modelType = "dnn"#"linear"
    epsList = [0.5]#, 0.0001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
else:
    exit(-1)

for algo in algos:
    results = []
    seed = 0
    for i in range(n_modes):
        results.append(dict())
    keepreading = True
    while keepreading:
        for mode in range(n_modes):
            for epsilon in epsList:
                try:
                    data = pd.read_csv('./%s/%s_%s_%s_mode%d_eps%f_seed%d%s.csv' %(folder, dataset, metric, modelType, mode, epsilon, seed, algo))
                    #print("read file ", './%s/%s_%s_%s_mode%d_eps%f_seed%d%s.csv' %(folder, dataset, metric, modelType, mode, epsilon, seed, algo))
                    values = [data.values[nb_models+1][0], data.values[nb_models+1][1], data.values[nb_models+1][2], data.values[nb_models+1][3]] 
                    if not epsilon in results[mode]:
                        results[mode][epsilon] = []
                    results[mode][epsilon].append(values)
                    nb_read += 1
                    #print("mode %d, seed %d, epsilon=%f, read averages = " %(mode, seed, epsilon), values)
                except FileNotFoundError:
                    print("cannot open file: ", './%s/%s_%s_%s_mode%d_eps%f_seed%d%s.csv' %(folder, dataset, metric, modelType, mode, epsilon, seed, algo))
                    if seed >= nb_seed:
                        keepreading = False
                        break
                    else:
                        #seed+=1
                        continue
        if keepreading:
            seed+=1
    print("Read %d files." %nb_read)

    with open('./summary/%s_%s_%s_summary_%dmodels%dseeds%s.csv' %(dataset, metric, modelType, nb_models, seed, algo), mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Mode', 'epsilon', 'Training error', 'Training violation', 'Test error', 'Test violation', 'Training error STD', 'Training violation STD', 'Test error STD', 'Test violation STD', 'nbseeds'])
        for epsilon in epsList:
            for mode in range(n_modes):   
                nbSeeds = len(results[mode][epsilon])
                train_error = np.average([results[mode][epsilon][i][0] for i in range(nbSeeds)])
                train_violation = np.average([results[mode][epsilon][i][1] for i in range(nbSeeds)])
                test_error = np.average([results[mode][epsilon][i][2] for i in range(nbSeeds)])
                test_violation = np.average([results[mode][epsilon][i][3] for i in range(nbSeeds)])
                train_error_std = np.std([results[mode][epsilon][i][0] for i in range(nbSeeds)])
                train_violation_std = np.std([results[mode][epsilon][i][1] for i in range(nbSeeds)])
                test_error_std = np.std([results[mode][epsilon][i][2] for i in range(nbSeeds)])
                test_violation_std = np.std([results[mode][epsilon][i][3] for i in range(nbSeeds)])
                print("Mode ", mode, ", epsilon=", epsilon, " perfs (average for ", nbSeeds, " seeds) : train_error = ", train_error, ", train violation = ", train_violation, ", test error = ", test_error, " test violation = ", test_violation) 
                csv_writer.writerow([mode, epsilon, train_error, train_violation, test_error, test_violation, train_error_std, train_violation_std, test_error_std, test_violation_std, nbSeeds])

    with open('./summary/rounded_%s_%s_%s_summary_%dmodels%dseeds%s.csv' %(dataset, metric, modelType, nb_models, seed, algo), mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Mode', 'epsilon', 'Training error', 'Training violation', 'Test error', 'Test violation', 'Training error STD', 'Training violation STD', 'Test error STD', 'Test violation STD', 'nbseeds'])
        for epsilon in epsList:
            for mode in range(n_modes):   
                nbSeeds = len(results[mode][epsilon])
                n_digits_error = 3
                train_error = round(np.average([results[mode][epsilon][i][0] for i in range(nbSeeds)]), n_digits_error)
                train_violation = round(np.average([results[mode][epsilon][i][1] for i in range(nbSeeds)]), n_digits_violation)
                test_error = round(np.average([results[mode][epsilon][i][2] for i in range(nbSeeds)]), n_digits_error)
                test_violation = round(np.average([results[mode][epsilon][i][3] for i in range(nbSeeds)]), n_digits_violation)
                train_error_std = round(np.std([results[mode][epsilon][i][0] for i in range(nbSeeds)]), n_digits_violation+1)
                train_violation_std = round(np.std([results[mode][epsilon][i][1] for i in range(nbSeeds)]), n_digits_violation+1)
                test_error_std = round(np.std([results[mode][epsilon][i][2] for i in range(nbSeeds)]), n_digits_violation+1)
                test_violation_std = round(np.std([results[mode][epsilon][i][3] for i in range(nbSeeds)]), n_digits_violation+1)
                print("Mode ", mode, ", epsilon=", epsilon, " perfs (average for ", nbSeeds, " seeds) : train_error = ", train_error, ", train violation = ", train_violation, ", test error = ", test_error, " test violation = ", test_violation) 
                csv_writer.writerow([mode, epsilon, train_error, train_violation, test_error, test_violation, train_error_std, train_violation_std, test_error_std, test_violation_std, nbSeeds])

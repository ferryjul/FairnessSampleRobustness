import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Principe de la vérification de cohérence des résultats pour EOdds et CUAE :
# fact: somme des abs() < epsilon plus contraignant que max des abs() < epsilon
# car si somme des abs() < epsilon alors max des abs() < epsilon
# mais si max des abs() < epsilon alors pas forcément somme des abs() < epsilon
# donc en train on vérifie que pour chaque epsilon la meilleure valeur de fonction objectif atteinte est
# meilleure pour max() que pour somme !
metrics_str = ["Equalized Odds", "CUAE"]
metrics = [5,6]
masks = [0,10,30]
path = "./results-test"
path_old_expes = "../../../data"
dataset = "compas"
suffix="without_filtering"
heuristic = "BFS-objective-aware"
display=False
epsilon_range1 = np.arange(0.705, 0.901, 0.005)
epsilon_range2 = np.arange(0.902, 0.98, 0.002)
epsilon_range3 = np.arange(0.98, 0.9895, 0.001)
epsilon_range4 = np.arange(0.99, 1.000, 0.0002)
epsilon_range = list(epsilon_range1) + list(epsilon_range2) + list(epsilon_range3) + list(epsilon_range4)
base = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]
epsilon_range = base + epsilon_range
epsilons = [round(x,4) for x in epsilon_range] #147 values

for i, metric in enumerate(metrics):
    print("Verifying results for metric ", metric, "(%s)" %metrics_str[i])
    for m in masks:
        globalNewAccs=[]
        globalOldAccs=[]
        epsilons_read=[]
        for epsilon in epsilons:
            try:
                fileName='%s_eps%f_metric%d_%s_%s_%dmasks.csv' %(dataset, epsilon, metric, heuristic, suffix, m)
                content_new_expes = pd.read_csv('%s/%s' %(path, fileName))
                epsilons_read.append(epsilon)
                train_accs_new=[]
                train_unfs_new=[]
                obj_fcts_new=[]
                for l in range(len(content_new_expes.values)):
                    #print(content_new_expes.values[l][1])
                    train_accs_new.append(content_new_expes.values[l][1])
                    train_unfs_new.append(content_new_expes.values[l][2])
                    obj_fcts_new.append(content_new_expes.values[l][3])
                content_old_expes = pd.read_csv('%s/%s' %(path_old_expes, fileName))
                train_accs_old=[]
                train_unfs_old=[]
                obj_fcts_old=[]
                for l in range(len(content_old_expes.values)):
                    #print(content_new_expes.values[l][1])
                    train_accs_old.append(content_old_expes.values[l][1])
                    train_unfs_old.append(content_old_expes.values[l][2])
                    obj_fcts_old.append(content_old_expes.values[l][3])
                for j in range(len(obj_fcts_new)):
                    if obj_fcts_new[j]>obj_fcts_old[j]:
                        print("error: better objective function value (", obj_fcts_old[j], ") for old expes than for new ones (", obj_fcts_new[j], ") for fold ", j, "(with %d masks)" %m)
                        print("epsilon=", epsilon)
                        exit()
                globalNewAccs.append(train_accs_new[-1])
                globalOldAccs.append(train_accs_old[-1])
            except FileNotFoundError:
                print("Missing file ", fileName)
                continue
        if display:
            plt.plot(epsilons_read, globalNewAccs, label="new (%d masks)" %m)
            plt.plot(epsilons_read, globalOldAccs, label="old (%d masks)" %m)
            plt.title("dataset %s, %s" %(dataset, metrics_str[i]))
            plt.legend(loc='best')
            plt.show()
        print("No incoherence found for %d masks." %m)
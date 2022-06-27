import pandas as pd 


masks_map = {
    "no mask"   : "0masks",
    "10 masks"  : "10masks",
    "30 masks"  : "30masks"
}


def merge_pareto(partition = "train"):
    for dataset in ["adult", "compas", "default_credit", "marketing"]:
        for metric in [1, 2, 3, 4, 5, 6]:
            df_all = pd.DataFrame()
            for nbr_mask in ["no mask", "10 masks", "30 masks"]:
                filename = "./results_compiled/{}_DRO_compile_eps_metric{}_BFS-objective-aware_pareto_{}_without_filtering_{}.csv".format(dataset, metric, partition, masks_map[nbr_mask])
                df = pd.read_csv(filename)
                df["error_{}".format(partition)] = 1 - df["acc_{}".format(partition)]
                df["masks"] = nbr_mask
                df_all = pd.concat([df_all, df])
            output_filename = "./results_merged/{}_{}_{}.csv".format(dataset, partition, metric)
            df_all.to_csv(output_filename, index=False)

def merge_fairness_violation():
    for dataset in ["adult", "compas", "default_credit", "marketing"]:
        for metric in [1, 2, 3, 4, 5, 6]:
            filename = "./results_compiled/{}_DRO_compile_eps_metric{}_BFS-objective-aware_unf_violation_without_filtering.csv".format(dataset, metric)
            df = pd.read_csv(filename)
            df_list = []
            for nbr_mask in ["no mask", "10 masks", "30 masks"]:
                df_current = pd.DataFrame()
                df_current["unf_train"] = df["unf_train_{}".format(masks_map[nbr_mask])]
                df_current["unf_test"] = df["unf_test_{}".format(masks_map[nbr_mask])]
                df_current["masks"] = nbr_mask
                df_list.append(df_current)

            df_all = pd.concat(df_list)
            output_filename = "./results_merged/fairness_violation_{}_{}.csv".format(dataset, metric)
            df_all.to_csv(output_filename, index=False)


            



merge_pareto("train")
merge_pareto("test")

merge_fairness_violation()
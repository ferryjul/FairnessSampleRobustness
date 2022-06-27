

def get_data(id):
    dataset, decision, prediction_name, min_feature, min_pos, maj_feature, maj_pos = None, None, None, None, None, None, None

    if id==1:
        dataset = "adult"
        decision = "income"
        prediction_name="[income:>50K]"
        min_feature = "gender_Female"
        min_pos = 0
        maj_feature = "gender_Male"
        maj_pos = 1

    if id==2:
        dataset = "compas"
        decision = "two_year_recid"
        prediction_name="[two_year_recid]"
        min_feature = "race_African-American"
        min_pos = 0
        maj_feature = "race_Caucasian"
        maj_pos = 1

    if id==3:
        dataset = "marketing"
        decision = "subscribed"
        prediction_name="[subscribed]"
        min_feature = "age_age:30-60"
        min_pos = 0
        maj_feature = "age_age:not30-60"
        maj_pos = 1

    if id==4:
        dataset = "default_credit"
        decision = "DEFAULT_PAYEMENT"
        prediction_name="[default_payment_next_month]"
        min_feature = "SEX_Female"
        min_pos = 0
        maj_feature = "SEX_Male"
        maj_pos = 1

    return dataset, decision, prediction_name, min_feature, min_pos, maj_feature, maj_pos

def get_metric(metric):
    metrics = {
    1: "statistical_parity",
    2 : "predictive_parity",
    3 : "predictive_equality",
    4 : "equal_opportunity",
    5 : "equalized_odds",
    6 : "conditional_use_accuracy_equality"
    }
    return metrics[metric]

def get_strategy(strat):
    strategy, bfsMode, strategy_name = None, None, None

    if strat == 1:
        strategy, bfsMode, strategy_name = "bfs", 0, "bfs"

    if strat == 2:
        strategy, bfsMode, strategy_name = "curious", 0, "curious"

    if strat == 3:
        strategy, bfsMode, strategy_name = "lower_bound", 0, "lower_bound"


    if strat == 4:
        strategy, bfsMode, strategy_name = "bfs", 2, "bfs_objective_aware"

    return strategy, bfsMode, strategy_name

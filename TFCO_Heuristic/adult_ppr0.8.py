import argparse
import csv
import math
import random
import numpy as np
import pandas as pd
import warnings
from six.moves import xrange
import tensorflow.compat.v1 as tf
import tensorflow_constrained_optimization as tfco
import matplotlib.pyplot as plt

tf.disable_eager_execution()
warnings.filterwarnings('ignore')


# parser initialization
parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--epsilon', type=float, default=0.8, help='epsilon value (min fairness acceptable) for epsilon-constrained method')
parser.add_argument('--drovalseed', type=int, default=0, help='seed value for dro or validation splits')
parser.add_argument('--expe', type=int, default=-1, help='for runs with slurm')
parser.add_argument('--algo', type=int, default=3, help='either 3 or 4: the algorithm used, as in the paper of Cotter et Al. (2019) - 3 uses the ProxyLagrangianOptimizer while 4 uses the LagrangianOptimizer')
# Modes :
# 0 : unconstrained
# 1 : base
# 2 : val
# 3, 4, 5: DRo with 10, 30, 50 masks
parser.add_argument('--mode', type=int, default=0, help='between 0 and 5 (both included)')

args = parser.parse_args()
algo = args.algo
expe = args.expe
if expe == -1: # single run
    epsilon = args.epsilon
    drovalseed=args.drovalseed
else:
    epsilonList=[0, 0.0001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    epsilon = epsilonList[expe % 10]

    drovalseed=expe - (expe % 10) # Même seed pour toutes les valeurs d'epsilon

mode = args.mode

if mode == 3:
    nb_masks = 10
elif mode == 4:
    nb_masks = 30
elif mode == 5:
    nb_masks = 50

# LOADING AND PREPARING DATASET
# Adult (version notebook no_generalization)
CATEGORICAL_COLUMNS = [
    'workclass', 'education', 'marital_status', 'occupation', 'relationship',
    'race', 'gender', 'native_country'
]
CONTINUOUS_COLUMNS = [
    'age', 'capital_gain', 'capital_loss', 'hours_per_week', 'education_num'
]
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]
LABEL_COLUMN = 'label'

PROTECTED_COLUMNS = [
    'gender_Female', 'gender_Male', 'race_White', 'race_Black'
]
DRO_MASKS_COLUMNS = []

def get_data():
    train_df_raw = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", names=COLUMNS, skipinitialspace=True)
    test_df_raw = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", names=COLUMNS, skipinitialspace=True, skiprows=1)

    train_df_raw[LABEL_COLUMN] = (train_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
    test_df_raw[LABEL_COLUMN] = (test_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
    # Preprocessing Features
    pd.options.mode.chained_assignment = None  # default='warn'

    # Functions for preprocessing categorical and continuous columns.
    def binarize_categorical_columns(input_train_df, input_test_df, categorical_columns=[]):

        def fix_columns(input_train_df, input_test_df):
            test_df_missing_cols = set(input_train_df.columns) - set(input_test_df.columns)
            for c in test_df_missing_cols:
                input_test_df[c] = 0
                train_df_missing_cols = set(input_test_df.columns) - set(input_train_df.columns)
            for c in train_df_missing_cols:
                input_train_df[c] = 0
                input_train_df = input_train_df[input_test_df.columns]
            return input_train_df, input_test_df

        # Binarize categorical columns.
        binarized_train_df = pd.get_dummies(input_train_df, columns=categorical_columns)
        binarized_test_df = pd.get_dummies(input_test_df, columns=categorical_columns)
        # Make sure the train and test dataframes have the same binarized columns.
        fixed_train_df, fixed_test_df = fix_columns(binarized_train_df, binarized_test_df)
        return fixed_train_df, fixed_test_df

    def bucketize_continuous_column(input_train_df,
                                  input_test_df,
                                  continuous_column_name,
                                  num_quantiles=None,
                                  bins=None):
        assert (num_quantiles is None or bins is None)
        if num_quantiles is not None:
            train_quantized, bins_quantized = pd.qcut(
              input_train_df[continuous_column_name],
              num_quantiles,
              retbins=True,
              labels=False)
            input_train_df[continuous_column_name] = pd.cut(
              input_train_df[continuous_column_name], bins_quantized, labels=False)
            input_test_df[continuous_column_name] = pd.cut(
              input_test_df[continuous_column_name], bins_quantized, labels=False)
        elif bins is not None:
            input_train_df[continuous_column_name] = pd.cut(
              input_train_df[continuous_column_name], bins, labels=False)
            input_test_df[continuous_column_name] = pd.cut(
              input_test_df[continuous_column_name], bins, labels=False)

    # Filter out all columns except the ones specified.
    train_df = train_df_raw[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + [LABEL_COLUMN]]
    test_df = test_df_raw[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + [LABEL_COLUMN]]
    
    # Bucketize continuous columns.
    bucketize_continuous_column(train_df, test_df, 'age', num_quantiles=4)
    bucketize_continuous_column(train_df, test_df, 'capital_gain', bins=[-1, 1, 4000, 10000, 100000])
    bucketize_continuous_column(train_df, test_df, 'capital_loss', bins=[-1, 1, 1800, 1950, 4500])
    bucketize_continuous_column(train_df, test_df, 'hours_per_week', bins=[0, 39, 41, 50, 100])
    bucketize_continuous_column(train_df, test_df, 'education_num', bins=[0, 8, 9, 11, 16])
    train_df, test_df = binarize_categorical_columns(train_df, test_df, categorical_columns=CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS)
    feature_names = list(train_df.keys())
    feature_names.remove(LABEL_COLUMN)
    num_features = len(feature_names)
    
    return train_df, test_df, feature_names

train_df, test_df, FEATURE_NAMES = get_data()

# ** new -----------------
# masks addition (useful only for training set - but we must provide some as well for test set for shape reasons)

np.random.seed(drovalseed)

if mode == 3 or mode == 4 or mode == 5:
    for i in range(nb_masks):
        random_binary_vect = np.random.randint(0, 2, train_df.shape[0])
        #print(np.unique(random_binary_vect, return_counts=True))
        train_df['dro_mask_%d'%i] = random_binary_vect
        DRO_MASKS_COLUMNS.append('dro_mask_%d'%i)
        test_df['dro_mask_%d'%i] = np.ones(shape=[test_df.shape[0]])
# ** ---------------------

# create splits for test/val
if mode == 2:
    train_df['SPLIT_0'] = np.random.randint(0, 2, train_df.shape[0])
    train_df['SPLIT_1'] = train_df['SPLIT_0'].apply(lambda row: 1 - row)


# Model with masks (modified by ourselves)
def _construct_model(model_name, input_tensor, hidden_units=10):
    with tf.variable_scope('model_name', reuse=True):
        hidden = tf.layers.dense(
            inputs=input_tensor,
            units=hidden_units,
            activation=tf.nn.relu,
            reuse=tf.AUTO_REUSE,
            name=model_name + "_hidden")
        output = tf.layers.dense(
            inputs=hidden,
            units=1,
            activation=None,
            reuse=tf.AUTO_REUSE,
            name=model_name + "_outputs")
        '''output = tf.layers.dense(
            inputs=input_tensor,
            units=1,
            activation=None,
            reuse=tf.AUTO_REUSE,
            name=model_name + "_outputs")'''
        return output

class Model(object):
    def __init__(self,
                 model_name,
                feature_names,
                hidden_units=10,
                gen_split=False,
                ppr_max_ratio=0.8,
                useDRO=False,
                nb_masks=0):
        tf.random.set_random_seed(123)
        self.feature_names = feature_names
        self.ppr_max_ratio = ppr_max_ratio
        
        num_features = len(self.feature_names)
        self.gen_split = gen_split
        # ** new **
        self.useDRO = useDRO
        self.nb_masks = nb_masks
        # ---------
        if self.gen_split:
            self.features_split_0 = tf.placeholder(
                tf.float32, shape=(None, num_features), name='split_0_features_placeholder')
            self.features_split_1 = tf.placeholder(
                tf.float32, shape=(None, num_features), name='split_1_features_placeholder')
            self.split_0_labels = tf.placeholder(
                tf.float32, shape=(None, 1), name='split_0_labels_placeholder')
            self.split_1_labels = tf.placeholder(
                tf.float32, shape=(None, 1), name='split_1_labels_placeholder')
            self.split_0_predictions = _construct_model(
                model_name, self.features_split_0, hidden_units=hidden_units)
            self.split_1_predictions = _construct_model(
                model_name, self.features_split_1, hidden_units=hidden_units)
            self.protected_split_0 = [tf.placeholder(tf.float32, shape=(None, 1), name=attribute+"_placeholder0") for attribute in PROTECTED_COLUMNS]
            self.protected_split_1 = [tf.placeholder(tf.float32, shape=(None, 1), name=attribute+"_placeholder1") for attribute in PROTECTED_COLUMNS]


        self.features_placeholder = tf.placeholder(
            tf.float32, shape=(None, num_features), name='features_placeholder')
        self.labels_placeholder = tf.placeholder(
            tf.float32, shape=(None, 1), name='labels_placeholder')
        # ** new **
        if self.useDRO:
            self.dro_masks_placeholders = []
            for i in range(self.nb_masks):
                self.dro_masks_placeholders.append(tf.placeholder(
                tf.float32, shape=(None, 1), name='dro_masks_placeholder_%d' %i))
        # ---------
        self.predictions_tensor = _construct_model(
            model_name, self.features_placeholder, hidden_units=hidden_units)
        self.protected_placeholders = [tf.placeholder(tf.float32, shape=(None, 1), name=attribute+"_placeholder") for attribute in PROTECTED_COLUMNS]

    def build_train_op(self,
                       learning_rate,
                       unconstrained=False):
        if self.gen_split:
            ctx = tfco.split_rate_context(self.split_0_predictions, self.split_1_predictions, self.split_0_labels, self.split_1_labels)
            #positive_slice = ctx.subset(self.split_0_labels > 0, self.split_1_labels > 0) 
        else:
            ctx = tfco.rate_context(self.predictions_tensor, self.labels_placeholder)
            #positive_slice = ctx.subset(self.labels_placeholder > 0) 
            
        overall_ppr = tfco.positive_prediction_rate(ctx)#(positive_slice)
        constraints = []
        if not unconstrained:
            for i in range(len(PROTECTED_COLUMNS)):
                if self.gen_split:
                    slice_ppr = tfco.positive_prediction_rate(
                        ctx.subset(
                            (self.protected_split_0[i] > 0),
                            (self.protected_split_1[i] > 0)))

                else:
                    slice_ppr = tfco.positive_prediction_rate(
                        ctx.subset((self.protected_placeholders[i] > 0)))
                constraints.append(slice_ppr >= self.ppr_max_ratio * overall_ppr)
        # ** new **
        # add dro-related constraints (same thing as above but filtered by masks)
        if self.useDRO:
            for mask_id in range(self.nb_masks):
                # Note that self.gen_split must be False so I removed the corresponding part here
                slice_mask = ctx.subset((self.dro_masks_placeholders[mask_id]>0))

                overall_ppr_mask = tfco.positive_prediction_rate(slice_mask)
                if not unconstrained:
                    for i in range(len(PROTECTED_COLUMNS)):
                        slice_ppr_mask = tfco.positive_prediction_rate(
                                ctx.subset((self.protected_placeholders[i] > 0) & (self.dro_masks_placeholders[mask_id]>0)))
                        constraints.append(slice_ppr_mask >= self.ppr_max_ratio * overall_ppr_mask)
        # ---------
        error = tfco.error_rate(ctx)
        mp = tfco.RateMinimizationProblem(error, constraints)
        if algo == 3:
            opt = tfco.ProxyLagrangianOptimizerV1(tf.train.AdamOptimizer(learning_rate))
        elif algo == 4:
            opt = tfco.LagrangianOptimizerV1(tf.train.AdamOptimizer(learning_rate))
        else:
            print("Unknown algorithm parameter: ", algo)
        self.train_op = opt.minimize(mp)
        return self.train_op
  
    def feed_dict_helper(self, dataframe, train=False):
        feed_dict = {}
        if self.gen_split and train:
            feed_dict[self.features_split_0] = dataframe[dataframe['SPLIT_0'] > 0][self.feature_names]
            feed_dict[self.features_split_1] = dataframe[dataframe['SPLIT_1'] > 0][self.feature_names]
            feed_dict[self.split_0_labels] = dataframe[dataframe['SPLIT_0'] > 0][[LABEL_COLUMN]]
            feed_dict[self.split_1_labels] = dataframe[dataframe['SPLIT_1'] > 0][[LABEL_COLUMN]]
            for i, protected_attribute in enumerate(PROTECTED_COLUMNS):
                feed_dict[self.protected_split_0[i]] = dataframe[dataframe['SPLIT_0'] > 0][[protected_attribute]]
                feed_dict[self.protected_split_1[i]] = dataframe[dataframe['SPLIT_1'] > 0][[protected_attribute]]

        elif self.gen_split and not train:
            feed_dict[self.features_split_0] = dataframe[self.feature_names]
            feed_dict[self.features_split_1] = dataframe[self.feature_names]
            feed_dict[self.split_0_labels] = dataframe[[LABEL_COLUMN]]
            feed_dict[self.split_1_labels] = dataframe[[LABEL_COLUMN]]
            for i, protected_attribute in enumerate(PROTECTED_COLUMNS):
                feed_dict[self.protected_split_0[i]] = dataframe[[protected_attribute]]
                feed_dict[self.protected_split_1[i]] = dataframe[[protected_attribute]]

        feed_dict[self.features_placeholder] = dataframe[self.feature_names]
        feed_dict[self.labels_placeholder] = dataframe[[LABEL_COLUMN]]
        # ** new **
        if self.useDRO:
            for i in range(self.nb_masks):
                feed_dict[self.dro_masks_placeholders[i]] = dataframe[[DRO_MASKS_COLUMNS[i]]]
        # ---------
        
        for i, protected_attribute in enumerate(PROTECTED_COLUMNS):
            feed_dict[self.protected_placeholders[i]] = dataframe[[protected_attribute]]
            
        return feed_dict

def training_generator(model,
                       train_df,
                       test_df,
                       minibatch_size,
                       num_iterations_per_loop=1,
                       num_loops=1,
                       random_seed=31337):
    random.seed(random_seed)
    num_rows = train_df.shape[0]
    minibatch_size = min(minibatch_size, num_rows)
    permutation = list(range(train_df.shape[0]))
    random.shuffle(permutation)

    session = tf.Session()
    session.run((tf.global_variables_initializer(),
               tf.local_variables_initializer()))

    minibatch_start_index = 0
    for n in xrange(num_loops):
        for _ in xrange(num_iterations_per_loop):
            minibatch_indices = []
            while len(minibatch_indices) < minibatch_size:
                minibatch_end_index = (
                minibatch_start_index + minibatch_size - len(minibatch_indices))
                if minibatch_end_index >= num_rows:
                    minibatch_indices += range(minibatch_start_index, num_rows)
                    minibatch_start_index = 0
                else:
                    minibatch_indices += range(minibatch_start_index, minibatch_end_index)
                    minibatch_start_index = minibatch_end_index

            session.run(
                  model.train_op,
                  feed_dict=model.feed_dict_helper(
                      train_df.iloc[[permutation[ii] for ii in minibatch_indices]], train=True))
            
        train_predictions = session.run(
            model.predictions_tensor,
            feed_dict=model.feed_dict_helper(train_df))
        test_predictions = session.run(
            model.predictions_tensor,
            feed_dict=model.feed_dict_helper(test_df))

        yield (train_predictions, test_predictions)


def error_rate(predictions, labels):
    signed_labels = (
      (labels > 0).astype(np.float32) - (labels <= 0).astype(np.float32))
    numerator = (np.multiply(signed_labels, predictions) <= 0).sum()
    denominator = predictions.shape[0]
    return float(numerator) / float(denominator)

def positive_prediction_rate(predictions, subset):
    numerator = np.multiply((predictions > 0).astype(np.float32),
                          (subset > 0).astype(np.float32)).sum()
    denominator = (subset > 0).sum()
    return float(numerator) / float(denominator)

def ppr(df):
    """Measure the positive prediction rate."""
    fp = sum((df['predictions'] >= 0.0))
    #print("df shape = ", df.shape[0])
    return float(fp) / df.shape[0]

def _get_error_rate_and_constraints(df, ppr_max_ratio):
    """Computes the error and fairness violations."""
    error_rate_local = error_rate(df[['predictions']], df[[LABEL_COLUMN]])
    overall_ppr = ppr(df)
    return error_rate_local, overall_ppr, [(overall_ppr * ppr_max_ratio) - ppr(df[df[protected_attribute] > 0.5]) for protected_attribute in PROTECTED_COLUMNS]
                                            
def _get_exp_error_rate_constraints(cand_dist, error_rates_vector,
                                    overall_tpr_vector, constraints_matrix):
    expected_error_rate = np.dot(cand_dist, error_rates_vector)
    expected_overall_tpr = np.dot(cand_dist, overall_tpr_vector)
    expected_constraints = np.matmul(cand_dist, constraints_matrix)
    return expected_error_rate, expected_overall_tpr, expected_constraints


def get_iterate_metrics(cand_dist, best_cand_index, error_rate_vector,
                        overall_tpr_vector, constraints_matrix):
    metrics = {}
    exp_error_rate, exp_overall_tpr, exp_constraints = _get_exp_error_rate_constraints(
      cand_dist, error_rate_vector, overall_tpr_vector, constraints_matrix)
    metrics['m_stochastic_error_rate'] = exp_error_rate
    metrics['m_stochastic_overall_tpr'] = exp_overall_tpr
    metrics['m_stochastic_max_constraint_violation'] = max(exp_constraints)
    for i, constraint in enumerate(exp_constraints):
        metrics['m_stochastic_constraint_violation_%d' % i] = constraint
    metrics['best_error_rate'] = error_rate_vector[best_cand_index]
    metrics['last_error_rate'] = error_rate_vector[-1]
    metrics['t_stochastic_error_rate'] = sum(error_rate_vector) / len(
      error_rate_vector)
    metrics['best_overall_tpr'] = overall_tpr_vector[best_cand_index]
    metrics['last_overall_tpr'] = overall_tpr_vector[-1]
    metrics['t_stochastic_overall_tpr'] = sum(overall_tpr_vector) / len(
      overall_tpr_vector)
    avg_constraints = []
    best_constraints = []
    last_constraints = []
    for constraint_iterates in np.transpose(constraints_matrix):
        avg_constraint = sum(constraint_iterates) / len(constraint_iterates)
        avg_constraints.append(avg_constraint)
        best_constraints.append(constraint_iterates[best_cand_index])
        last_constraints.append(constraint_iterates[-1])
    metrics['best_max_constraint_violation'] = max(best_constraints)
    for i, constraint in enumerate(best_constraints):
        metrics['best_constraint_violation_%d' % i] = constraint
    metrics['last_max_constraint_violation'] = max(last_constraints)
    for i, constraint in enumerate(last_constraints):
        metrics['last_constraint_violation_%d' % i] = constraint
    metrics['t_stochastic_max_constraint_violation'] = max(avg_constraints)
    for i, constraint in enumerate(avg_constraints):
        metrics['t_stochastic_constraint_violation_%d' % i] = constraint
    metrics['all_errors'] = error_rate_vector
    metrics['all_violations'] = np.max(constraints_matrix, axis=1)
    return metrics

def training_helper(model,
                    train_df,
                    test_df,
                    minibatch_size,
                    num_iterations_per_loop=1,
                    num_loops=1,
                    random_seed_arg=31337):
    train_objective_vector = []
    train_constraints_loss_matrix = []
    train_error_rate_vector = []
    train_overall_tpr_vector = []
    train_constraints_matrix = []
    test_error_rate_vector = []
    test_overall_tpr_vector = []
    test_constraints_matrix = []
    for train, test in training_generator(
        model, train_df, test_df, minibatch_size, num_iterations_per_loop, num_loops, random_seed=random_seed_arg):
        train_df['predictions'] = train
        test_df['predictions'] = test

        if model.gen_split:
            train_error_rate_split0, train_overall_tpr0, train_constraints_split0 = _get_error_rate_and_constraints(train_df[train_df['SPLIT_0'] > 0], model.ppr_max_ratio)
            train_error_rate_split1, train_overall_tpr1, train_constraints_split1 = _get_error_rate_and_constraints(train_df[train_df['SPLIT_1'] > 0], model.ppr_max_ratio)
            train_error_rate_vector.append(train_error_rate_split0)
            train_constraints_matrix.append(train_constraints_split1)
            train_constraints_loss_matrix.append(train_constraints_split1)
            train_overall_tpr_vector.append((train_overall_tpr0 + train_overall_tpr1) / 2)
        else:
            train_error_rate, train_overall_tpr, train_constraints = _get_error_rate_and_constraints(train_df, model.ppr_max_ratio)
            train_error_rate_vector.append(train_error_rate)
            train_overall_tpr_vector.append(train_overall_tpr)
            train_constraints_matrix.append(train_constraints)

        test_error_rate, test_overall_tpr, test_constraints = _get_error_rate_and_constraints(
            test_df, model.ppr_max_ratio)
        test_error_rate_vector.append(test_error_rate)
        test_overall_tpr_vector.append(test_overall_tpr)
        test_constraints_matrix.append(test_constraints)

    cand_dist = tfco.find_best_candidate_distribution(
      train_error_rate_vector, train_constraints_matrix, epsilon=0.001)
    best_cand_index = tfco.find_best_candidate_index(
      train_error_rate_vector, train_constraints_matrix)
    train_metrics = get_iterate_metrics(
      cand_dist, best_cand_index, train_error_rate_vector,
      train_overall_tpr_vector, train_constraints_matrix)
    test_metrics = get_iterate_metrics(
      cand_dist, best_cand_index, test_error_rate_vector,
      test_overall_tpr_vector, test_constraints_matrix)

    return (train_metrics, test_metrics)

trainErrors = []
testErrors = []
trainUnfs = []
testUnfs = []
n_models = 1
minibatch_size = 100
num_iterations_per_loop = int(np.ceil(train_df.shape[0]/minibatch_size))
#print("num_iterations_per_loop = ", num_iterations_per_loop)
num_loops = 40
nb_hidden = 50
print("hello")
if mode == 0:
    print("Running in unconstrained mode")
    # Train 10 models and compute average performances
    for model_id in range(n_models):
        model = Model("model_%d" %model_id, FEATURE_NAMES, hidden_units=nb_hidden, gen_split=False, useDRO=False,ppr_max_ratio=epsilon)
        model.build_train_op(0.01, unconstrained=True)
        results = training_helper(
            model,
            train_df,
            test_df,
            minibatch_size,
            num_iterations_per_loop=num_iterations_per_loop,
            num_loops=num_loops,
            random_seed_arg=drovalseed*2)
        trainErrors.append(results[0]["last_error_rate"])
        testErrors.append(results[1]["last_error_rate"])
        trainUnfs.append(results[0]["last_max_constraint_violation"])
        testUnfs.append(results[1]["last_max_constraint_violation"])
elif mode == 1:
    print("Running in single dataset mode")
    # Train 10 models and compute average performances
    for model_id in range(n_models):
        model = Model("model_%d" %model_id, FEATURE_NAMES, hidden_units=nb_hidden, gen_split=False, useDRO=False,ppr_max_ratio=epsilon)
        model.build_train_op(0.01, unconstrained=False)
        results = training_helper(
            model,
            train_df,
            test_df,
            minibatch_size,
            num_iterations_per_loop=num_iterations_per_loop,
            num_loops=num_loops,
            random_seed_arg=drovalseed*2)
        trainErrors.append(results[0]["m_stochastic_error_rate"])
        testErrors.append(results[1]["m_stochastic_error_rate"])
        trainUnfs.append(results[0]["m_stochastic_max_constraint_violation"])
        testUnfs.append(results[1]["m_stochastic_max_constraint_violation"])
elif mode == 2:
    print("Running in two-datasets mode")
    # Train 10 models and compute average performances
    for model_id in range(n_models):
        model = Model("model_%d" %model_id, FEATURE_NAMES, hidden_units=nb_hidden, gen_split=True, useDRO=False,ppr_max_ratio=epsilon)
        model.build_train_op(0.01, unconstrained=False)
        results = training_helper(
            model,
            train_df,
            test_df,
            minibatch_size,
            num_iterations_per_loop=num_iterations_per_loop,
            num_loops=num_loops)
        trainErrors.append(results[0]["m_stochastic_error_rate"])
        testErrors.append(results[1]["m_stochastic_error_rate"])
        trainUnfs.append(results[0]["m_stochastic_max_constraint_violation"])
        testUnfs.append(results[1]["m_stochastic_max_constraint_violation"])
elif mode == 3 or mode == 4 or mode == 5:
    print("Running in DRO-%dmasks mode" %nb_masks)
    # Train 10 models and compute average performances
    for model_id in range(n_models):
        model = Model("model_%d" %model_id, FEATURE_NAMES, hidden_units=nb_hidden, gen_split=False, useDRO=True,ppr_max_ratio=epsilon, nb_masks=nb_masks)
        model.build_train_op(0.01, unconstrained=False)
        results = training_helper(
            model,
            train_df,
            test_df,
            minibatch_size,
            num_iterations_per_loop=num_iterations_per_loop,
            num_loops=num_loops)
        trainErrors.append(results[0]["m_stochastic_error_rate"])
        testErrors.append(results[1]["m_stochastic_error_rate"])
        trainUnfs.append(results[0]["m_stochastic_max_constraint_violation"])
        testUnfs.append(results[1]["m_stochastic_max_constraint_violation"])
else:
    print("unknown mode. exiting.")
    exit()

    
with open('./runAdultPPR/adult_ppr_dnn50_mode%d_eps%f_seed%d_algo%d.csv' %(mode, epsilon, drovalseed, algo), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Training error', 'Training violation', 'Test error', 'Test violation'])
    print('Training error', 'Training violation', 'Test error', 'Test violation')
    for index in range(n_models):
        csv_writer.writerow([trainErrors[index], trainUnfs[index], testErrors[index], testUnfs[index]])
        print(trainErrors[index], trainUnfs[index], testErrors[index], testUnfs[index])
    csv_writer.writerow(['','','',''])
    csv_writer.writerow([np.average(trainErrors), np.average(trainUnfs), np.average(testErrors), np.average(testUnfs)])
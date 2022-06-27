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
parser.add_argument('--epsilon', type=float, default=0.00, help='epsilon value (min fairness acceptable) for epsilon-constrained method')
parser.add_argument('--drovalseed', type=int, default=0, help='seed value for dro or validation splits')
parser.add_argument('--expe', type=int, default=-1, help='for runs with slurm')
parser.add_argument('--algo', type=int, default=3, help='either 3 or 4: the algorithm used, as in the paper of Cotter et Al. (2019) - 3 uses the ProxyLagrangianOptimizer while 4 uses the LagrangianOptimizer')

# Modes :
# 0 : unconstrained
# 1 : base
# 2 : val
# 3, 4, 5: DRo with 10, 30, 50 masks
parser.add_argument('--mode', type=int, default=0, help='between 0 and 5 (both included)')

min_acceptable_tpr = 0.5
args = parser.parse_args()
modelType = "dnn" # "linear" or "dnn"
expe = args.expe


modelType = "dnn" # "linear" or "dnn"
if expe == -1: # single run
    epsilon = args.epsilon
    drovalseed=args.drovalseed
    mode = args.mode
    algo = args.algo
else:
    epsilon = args.epsilon
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    rank = COMM.Get_rank()
    drovalseed=rank
    cart_product = []
    modesList = [0, 1, 2, 3, 4, 5]
    algosList = [3, 4]
    for m in modesList:
        for a in algosList:
            cart_product.append([m, a])
    mode, algo = cart_product[expe]

if mode == 3:
    nb_masks = 10
elif mode == 4:
    nb_masks = 30
elif mode == 5:
    nb_masks = 50

# LOADING AND PREPARING DATASET
# Marketing
df = pd.read_csv("data/default of credit card clients.csv", na_values=['?'], sep=",")
#df = pd.read_csv("bank/default_credit_fullDiscretized.csv", na_values=['?'], sep=",")

LABEL_COLUMN = 'label'

CATEGORICAL_COLUMNS = [ 'sex_cat', 'age_cat']

PROTECTED_COLUMNS = ['sex_cat_male', 'sex_cat_female', 'age_cat_young', 'age_cat_old'] #, 'age_cat_very_young', 'age_cat_young', 'age_cat_medium', 'age_cat_old']#, 'age_cat_very_young', 'age_cat_young', 'age_cat_medium', 'age_cat_old']

CONTINUOUS_LABEL_COLUMN = 'default_payment_next_month'

BINARY_LABEL_COLUMN = 'label'

EXCLUDED_COLUMNS = ['default_payment_next_month', 'ID']

DRO_MASKS_COLUMNS = []

def _train_test_split(df, test_frac=0.33, seed=42): # 0.33, seed=42
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    n = len(df)

    test_end = int(test_frac * n)
    test_df = df.iloc[perm[:test_end]]
    train_df = df.iloc[perm[test_end:]]

    return train_df, test_df

def add_binary_label(input_df):
    input_df[LABEL_COLUMN] = input_df[CONTINUOUS_LABEL_COLUMN].map({1: 1,0: 0})
    
def add_protected_categories(input_df):
    #input_df['age_cat'] = pd.qcut(
    #  input_df['AGE'], 4, labels=['very_young', 'young', 'medium', 'old'])
    input_df['age_cat'] = pd.qcut(
      input_df['AGE'], 2, labels=['young', 'old'])
    # Gender (1 = male; 2 = female).
    input_df['sex_cat'] = input_df['SEX'].map({1: 'male',2: 'female'})


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

add_binary_label(df)

add_protected_categories(df) # add age categories (4)

#print(df)

df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS)



FEATURE_NAMES = [
      name for name in df.keys()
      if name not in [LABEL_COLUMN] + EXCLUDED_COLUMNS + PROTECTED_COLUMNS
]

#for f in FEATURE_NAMES:#df.columns:
#    print(df[f])

#print(df[PROTECTED_COLUMNS])

train_df, test_df = _train_test_split(df, seed=drovalseed)

# scale feature values to improve NN performances
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_df[FEATURE_NAMES] = sc.fit_transform(train_df[FEATURE_NAMES]) # Fit to data, then transform it.
test_df[FEATURE_NAMES] = sc.transform(test_df[FEATURE_NAMES]) # Perform standardization by centering and scaling

'''print("Train df (for example):")
print("-- protected groups:")
print(train_df[PROTECTED_COLUMNS])
print("-- label:")
print(train_df[LABEL_COLUMN])
print("-- features (for training):")
print(train_df[FEATURE_NAMES])'''


for column in FEATURE_NAMES:
    train_mean = train_df[column].mean()
    train_df[column].fillna(train_mean, inplace=True)
    test_df[column].fillna(train_mean, inplace=True)

#print(df)
# ** new -----------------
# masks addition (useful only for training set - but we must provide some as well for test set for shape reasons)
#print("Number of features = ", len(FEATURE_NAMES), "; training examples = ", train_df.shape[0], "; test examples = ", test_df.shape[0])
np.random.seed(drovalseed)

#for f in FEATURE_NAMES:#df.columns:
#    print(train_df[f])
exit(0)
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


# Model with masks (modified by julien)
def _construct_model(model_name, input_tensor, hidden_units=10):
    with tf.variable_scope('model_name', reuse=True):
        if modelType == "linear":
            output = tf.layers.dense(
            inputs=input_tensor,
            units=1,
            activation=None,
            reuse=tf.AUTO_REUSE,
            name=model_name + "_outputs")
        elif modelType == "dnn":
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
        return output

class Model(object):
    def __init__(self,
                 model_name,
                feature_names,
                hidden_units=10,
                gen_split=False,
                max_fpr=0,
                useDRO=False,
                nb_masks=0):
        tf.random.set_random_seed(123)
        self.feature_names = feature_names
        self.max_fpr = max_fpr
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
        constraints = []
        '''# FPR Constraints
        if self.gen_split:
            ctx = tfco.split_rate_context(self.split_0_predictions, self.split_1_predictions, self.split_0_labels, self.split_1_labels)
            negative_slice = ctx.subset(self.split_0_labels <= 0, self.split_1_labels <= 0) 
        else:
            ctx = tfco.rate_context(self.predictions_tensor, self.labels_placeholder)
            negative_slice = ctx.subset(self.labels_placeholder <= 0)
            
        overall_fpr = tfco.positive_prediction_rate(negative_slice)
        
        if not unconstrained:
            for i in range(len(PROTECTED_COLUMNS)):
                if self.gen_split:
                    slice_fpr = tfco.positive_prediction_rate(
                        ctx.subset(
                            (self.split_0_labels <= 0) & (self.protected_split_0[i] > 0),
                            (self.split_1_labels <= 0) & (self.protected_split_1[i] > 0)))

                else:
                    slice_fpr = tfco.positive_prediction_rate(
                        ctx.subset((self.protected_placeholders[i] > 0) & (self.labels_placeholder <= 0)))
                constraints.append(slice_fpr <= (self.max_fpr))
        # ** new **
        # add dro-related constraints (same thing as above but filtered by masks)
        if self.useDRO:
            for mask_id in range(self.nb_masks):
                # Note that self.gen_split must be False so I removed the corresponding part here
                negative_slice_mask = ctx.subset((self.labels_placeholder <= 0) & (self.dro_masks_placeholders[mask_id]>0))

                overall_fpr_mask = tfco.positive_prediction_rate(negative_slice_mask)
                if not unconstrained:
                    for i in range(len(PROTECTED_COLUMNS)):
                        slice_fpr_mask = tfco.positive_prediction_rate(
                                ctx.subset((self.protected_placeholders[i] > 0) & (self.labels_placeholder <= 0) & (self.dro_masks_placeholders[mask_id]>0)))
                        constraints.append(slice_fpr_mask <= (self.max_fpr))'''

        # TPR Constraints
        if self.gen_split:
            ctx = tfco.split_rate_context(self.split_0_predictions, self.split_1_predictions, self.split_0_labels, self.split_1_labels)
            positive_slice = ctx.subset(self.split_0_labels > 0, self.split_1_labels > 0) 
        else:
            ctx = tfco.rate_context(self.predictions_tensor, self.labels_placeholder)
            positive_slice = ctx.subset(self.labels_placeholder > 0) 
            
        overall_tpr = tfco.positive_prediction_rate(positive_slice)
        if not unconstrained:
            for i in range(len(PROTECTED_COLUMNS)):
                if self.gen_split:
                    slice_tpr = tfco.positive_prediction_rate(
                        ctx.subset(
                            (self.split_0_labels > 0) & (self.protected_split_0[i] > 0),
                            (self.split_1_labels > 0) & (self.protected_split_1[i] > 0)))

                else:
                    slice_tpr = tfco.positive_prediction_rate(
                        ctx.subset((self.protected_placeholders[i] > 0) & (self.labels_placeholder > 0)))
                constraints.append(slice_tpr >= min_acceptable_tpr)
        # ** new **
        # add dro-related constraints (same thing as above but filtered by masks)
        if not unconstrained and self.useDRO:
            for mask_id in range(self.nb_masks):
                # Note that self.gen_split must be False so I removed the corresponding part here
                positive_slice_mask = ctx.subset((self.labels_placeholder > 0) & (self.dro_masks_placeholders[mask_id]>0))

                overall_tpr_mask = tfco.positive_prediction_rate(positive_slice_mask)
                for i in range(len(PROTECTED_COLUMNS)):
                    slice_tpr_mask = tfco.positive_prediction_rate(
                            ctx.subset((self.protected_placeholders[i] > 0) & (self.labels_placeholder > 0) & (self.dro_masks_placeholders[mask_id]>0)))
                    constraints.append(slice_tpr_mask >= min_acceptable_tpr)
        

        # ---------
        print("built with #constraints = ", len(constraints))
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

def fpr(df):
    fp = sum((df['predictions'] >= 0.0) & (df[LABEL_COLUMN] < 0.5))
    ln = sum(df[LABEL_COLUMN] < 0.5)
    return float(fp) / float(ln)

def fnr(df):
    fn = sum((df['predictions'] < 0.0) & (df[LABEL_COLUMN] > 0.5))
    ln = sum(df[LABEL_COLUMN] > 0.5)
    return float(fn) / float(ln)

def tpr(df):
    """Measure the true positive rate."""
    fp = sum((df['predictions'] >= 0.0) & (df[LABEL_COLUMN] > 0.5))
    ln = sum(df[LABEL_COLUMN] > 0.5)
    return float(fp) / float(ln)

def _get_error_rate_and_constraints(df, max_fpr):
    error_rate_local = error_rate(df[['predictions']], df[[LABEL_COLUMN]])
    overall_fpr = fpr(df)
    #violations_all = [(fpr(df[df[protected_attribute] > 0.5]) - max_fpr) for protected_attribute in PROTECTED_COLUMNS]
    violations_all = [min_acceptable_tpr-(tpr(df[df[protected_attribute] > 0.5])) for protected_attribute in PROTECTED_COLUMNS]
    #print(violations_all)
    return error_rate_local, overall_fpr, violations_all

def _get_exp_error_rate_constraints(cand_dist, error_rates_vector,
                                    overall_fpr_vector, constraints_matrix):
    expected_error_rate = np.dot(cand_dist, error_rates_vector)
    expected_overall_fpr = np.dot(cand_dist, overall_fpr_vector)
    expected_constraints = np.matmul(cand_dist, constraints_matrix)
    return expected_error_rate, expected_overall_fpr, expected_constraints

def get_iterate_metrics(cand_dist, best_cand_index, error_rate_vector,
                        overall_fpr_vector, constraints_matrix):
    metrics = {}
    exp_error_rate, exp_overall_fpr, exp_constraints = _get_exp_error_rate_constraints(
      cand_dist, error_rate_vector, overall_fpr_vector, constraints_matrix)
    metrics['m_stochastic_error_rate'] = exp_error_rate
    metrics['m_stochastic_overall_fpr'] = exp_overall_fpr
    metrics['m_stochastic_max_constraint_violation'] = max(exp_constraints)
    for i, constraint in enumerate(exp_constraints):
        metrics['m_stochastic_constraint_violation_%d' % i] = constraint
    metrics['best_error_rate'] = error_rate_vector[best_cand_index]
    metrics['last_error_rate'] = error_rate_vector[-1]
    metrics['t_stochastic_error_rate'] = sum(error_rate_vector) / len(
      error_rate_vector)
    metrics['best_overall_fpr'] = overall_fpr_vector[best_cand_index]
    metrics['last_overall_fpr'] = overall_fpr_vector[-1]
    metrics['t_stochastic_overall_fpr'] = sum(overall_fpr_vector) / len(
      overall_fpr_vector)
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
    train_overall_fpr_vector = []
    train_constraints_matrix = []
    test_error_rate_vector = []
    test_overall_fpr_vector = []
    test_constraints_matrix = []
    for train, test in training_generator(
        model, train_df, test_df, minibatch_size, num_iterations_per_loop, num_loops, random_seed=random_seed_arg):
        train_df['predictions'] = train
        test_df['predictions'] = test

        if model.gen_split:
            train_error_rate_split0, train_overall_fpr0, train_constraints_split0 = _get_error_rate_and_constraints(train_df[train_df['SPLIT_0'] > 0], model.max_fpr)
            train_error_rate_split1, train_overall_fpr1, train_constraints_split1 = _get_error_rate_and_constraints(train_df[train_df['SPLIT_1'] > 0], model.max_fpr)
            train_error_rate_vector.append(train_error_rate_split0)
            train_constraints_matrix.append(train_constraints_split1)
            train_constraints_loss_matrix.append(train_constraints_split1)
            train_overall_fpr_vector.append((train_overall_fpr0 + train_overall_fpr1) / 2)
        else:
            train_error_rate, train_overall_fpr, train_constraints = _get_error_rate_and_constraints(train_df, model.max_fpr)
            train_error_rate_vector.append(train_error_rate)
            train_overall_fpr_vector.append(train_overall_fpr)
            train_constraints_matrix.append(train_constraints)

        test_error_rate, test_overall_fpr, test_constraints = _get_error_rate_and_constraints(
            test_df, model.max_fpr)
        test_error_rate_vector.append(test_error_rate)
        test_overall_fpr_vector.append(test_overall_fpr)
        test_constraints_matrix.append(test_constraints)

    cand_dist = tfco.find_best_candidate_distribution(
      train_error_rate_vector, train_constraints_matrix, epsilon=0.001)
    best_cand_index = tfco.find_best_candidate_index(
      train_error_rate_vector, train_constraints_matrix)
    train_metrics = get_iterate_metrics(
      cand_dist, best_cand_index, train_error_rate_vector,
      train_overall_fpr_vector, train_constraints_matrix)
    test_metrics = get_iterate_metrics(
      cand_dist, best_cand_index, test_error_rate_vector,
      test_overall_fpr_vector, test_constraints_matrix)

    return (train_metrics, test_metrics)

trainErrors = []
testErrors = []
trainUnfs = []
testUnfs = []
n_models = 1
minibatch_size = 100
num_iterations_per_loop = int(np.ceil(train_df.shape[0]/minibatch_size))#14
#print("num_iterations_per_loop=", num_iterations_per_loop)
num_loops = 100 #200
n_hidden = 50 # useful only if modelType=="dnn"
modePrint = False
if mode == 0:
    if modePrint:
        print("Running in unconstrained mode")
    # Train 10 models and compute average performances
    for model_id in range(n_models):
        model = Model("model_%d" %model_id, FEATURE_NAMES, hidden_units=n_hidden, gen_split=False, useDRO=False,max_fpr=epsilon)
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
    if modePrint:
        print("Running in single dataset mode")
    # Train 10 models and compute average performances
    for model_id in range(n_models):
        model = Model("model_%d" %model_id, FEATURE_NAMES, hidden_units=n_hidden, gen_split=False, useDRO=False,max_fpr=epsilon)
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
    if modePrint:
        print("Running in two-datasets mode")
    # Train 10 models and compute average performances
    for model_id in range(n_models):
        model = Model("model_%d" %model_id, FEATURE_NAMES, hidden_units=n_hidden, gen_split=True, useDRO=False,max_fpr=epsilon)
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
    if modePrint:
        print("Running in DRO-%dmasks mode" %nb_masks)
    # Train 10 models and compute average performances
    for model_id in range(n_models):
        model = Model("model_%d" %model_id, FEATURE_NAMES, hidden_units=n_hidden, gen_split=False, useDRO=True,max_fpr=epsilon, nb_masks=nb_masks)
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

if modePrint:
    print("---------------------------------------------------------------------")
with open('./default_credit_min_tpr_0.5/default_credit_min_tpr_%s_mode%d_eps%f_seed%d_algo%d.csv' %(modelType, mode, min_acceptable_tpr, drovalseed, algo), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Training error', 'Training violation', 'Test error', 'Test violation'])
    if modePrint:
        print('Training error', 'Training violation', 'Test error', 'Test violation')
    for index in range(n_models):
        csv_writer.writerow([trainErrors[index], trainUnfs[index], testErrors[index], testUnfs[index]])
        print(trainErrors[index], trainUnfs[index], testErrors[index], testUnfs[index])
    csv_writer.writerow(['','','',''])
    csv_writer.writerow([np.average(trainErrors), np.average(trainUnfs), np.average(testErrors), np.average(testUnfs)])
if modePrint:
    print("---------------------------------------------------------------------")
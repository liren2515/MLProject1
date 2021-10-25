
import os, sys
import time
import numpy as np
from lib.data import load_data, statistic_invalid, standardize, save_data, cross_validation, build_poly_with_interation_3, build_sin_2, build_poly_power, add_bias
from lib import config
from lib import method
from lib.plots import plot_train_test

config.data_balance = False
config.power_degree = 15
config.num_interval_lambda = 15


train_path_dataset = 'data/train.csv'
test_path_dataset = 'data/test.csv'
# If data_balance is true, it will replicate samoe examples from one class 
# to make two classes have the same number of samples.
# May or may not be helpful.
data_PRI_raw, data_DER_raw, prediction = load_data(train_path_dataset, balance=config.data_balance)
data_PRI_raw_test, data_DER_raw_test, _ = load_data(test_path_dataset)

num_sample = len(prediction)
num_DER = data_DER_raw.shape[-1]
num_PRI = data_PRI_raw.shape[-1]

# Exclude invalid features
data_DER_raw = data_DER_raw[:,config.valid_DER_idx] # 8-D
data_PRI_raw = data_PRI_raw[:,config.valid_PRI_idx] 

data_DER_raw_test = data_DER_raw_test[:,config.valid_DER_idx]
data_PRI_raw_test = data_PRI_raw_test[:,config.valid_PRI_idx]

# Feature augmentation
print('It will take some time to build the features of augmentation. Be patient.')
print('We run it on a server. Make sure you have enough CPU memory!')
# Sine augmentation 
data_DER_raw_sin2 = build_sin_2(data_DER_raw)
data_DER_raw_test_sin2 = build_sin_2(data_DER_raw_test)

data_PRI_raw_sin2 = build_sin_2(data_PRI_raw)
data_PRI_raw_test_sin2 = build_sin_2(data_PRI_raw_test)
print('Build sine down!')

# Power-series augmentation
data_DER_raw_power = build_poly_power(data_DER_raw, degree=config.power_degree)
data_DER_raw_test_power = build_poly_power(data_DER_raw_test, degree=config.power_degree)

data_PRI_raw_power = build_poly_power(data_PRI_raw, degree=config.power_degree)
data_PRI_raw_test_power = build_poly_power(data_PRI_raw_test, degree=config.power_degree)
print('Build power-series down!')

# Polynomial augmentation
data_DER_raw = build_poly_with_interation_3(data_DER_raw)
data_DER_raw_test = build_poly_with_interation_3(data_DER_raw_test)

data_PRI_raw = build_poly_with_interation_3(data_PRI_raw)
data_PRI_raw_test = build_poly_with_interation_3(data_PRI_raw_test)
print('Build poly down!')

# Concatenate features
data_DER_raw = np.concatenate((data_DER_raw, data_DER_raw_sin2, data_DER_raw_power), axis=-1)
data_DER_raw_test = np.concatenate((data_DER_raw_test, data_DER_raw_test_sin2, data_DER_raw_test_power), axis=-1)
data_PRI_raw = np.concatenate((data_PRI_raw, data_PRI_raw_sin2, data_PRI_raw_power), axis=-1)
data_PRI_raw_test = np.concatenate((data_PRI_raw_test, data_PRI_raw_test_sin2, data_PRI_raw_test_power), axis=-1)
print(data_DER_raw.shape, data_PRI_raw.shape)

# Data normalization
data_DER, mean_DER, std_DER = standardize(data_DER_raw)
data_DER_test = (data_DER_raw_test - mean_DER) / std_DER
data_DER = add_bias(data_DER)
data_DER_test = add_bias(data_DER_test)

data_PRI, mean_PRI, std_PRI = standardize(data_PRI_raw)
data_PRI_test = (data_PRI_raw_test - mean_PRI) / std_PRI
data_PRI = add_bias(data_PRI)
data_PRI_test = add_bias(data_PRI_test)

data_DER = np.concatenate((data_DER, data_PRI), axis=-1)
data_DER_test = np.concatenate((data_DER_test, data_PRI_test), axis=-1)


print('Run Ridge Regression!')
function = method.ridge_regression
y = prediction
tx = data_DER
lambdas = np.logspace(-16, -2, config.num_interval_lambda)

# Create sub-dataset for cross validation
num_samples = len(y)
ratio = 0.1
seed = 42
train_sets, val_sets = cross_validation(num_samples, ratio=ratio, seed=seed)
record_acc_train = []   
record_acc_val = []
record_w = []

# Run cross validation
for ind, lambda_ in enumerate(lambdas):
    acc_train, acc_val = [], []
    print('Evaluate lambda: ', lambda_)
    for train_set, val_set in zip(train_sets, val_sets):
        y_train, y_val = y[train_set], y[val_set]
        tx_train, tx_val = tx[train_set], tx[val_set]
        mse, w = function(y_train, tx_train, lambda_)
        prediction_train = tx_train.dot(w)
        prediction_train[prediction_train<0] = -1
        prediction_train[prediction_train>=0] = 1
        prediction_val = tx_val.dot(w)
        prediction_val[prediction_val<0] = -1
        prediction_val[prediction_val>=0] = 1

        acc_tr = (prediction_train == y_train).astype(float).sum()/len(y_train)
        acc_v = (prediction_val == y_val).astype(float).sum()/len(y_val)
        acc_train.append(acc_tr)
        acc_val.append(acc_v)
    record_acc_train.append(sum(acc_train)/len(acc_train))
    record_acc_val.append(sum(acc_val)/len(acc_val))

save_folder = 'results/best_result'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

fig_save_path = os.path.join(save_folder, 'ridge_regression_degree_%d.png'%config.power_degree) 
plot_train_test(record_acc_train, record_acc_val, lambdas, fig_save_path)

# Get best lambda
idx_max = np.argmax(np.array(record_acc_val))
lambda_ = lambdas[idx_max]

# Test
_, w = function(y, tx, lambda_)
prediction_test = data_DER_test.dot(w)
prediction_test[prediction_test<0] = -1
prediction_test[prediction_test>=0] = 1

# Save results
csv_save_path = os.path.join(save_folder, 'ridge_regression_degree_%d.csv'%config.power_degree)
save_data(prediction_test, csv_save_path)
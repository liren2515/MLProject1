import os
import numpy as np
from lib.data import load_data, standardize, save_data, cross_validation, build_poly_with_interation_3, build_sin_2, \
    build_poly_power, add_bias
from lib import config
from lib import method
from lib.plots import plot_train_test

train_path_dataset = 'data/train.csv'
test_path_dataset = 'data/test.csv'
# If data_balance is true, it will replicate same examples from one class
# to make two classes have the same number of samples.
# May or may not be helpful.
data_PRI_raw, data_DER_raw, prediction = load_data(train_path_dataset, balance=config.data_balance)
data_PRI_raw_test, data_DER_raw_test, _ = load_data(test_path_dataset)

num_sample = len(prediction)
num_DER = data_DER_raw.shape[-1]
num_PRI = data_PRI_raw.shape[-1]

# Exclude invalid features
data_DER_raw = data_DER_raw[:, config.valid_DER_idx]  # 8-D
data_PRI_raw = data_PRI_raw[:, config.valid_PRI_idx]

data_DER_raw_test = data_DER_raw_test[:, config.valid_DER_idx]
data_PRI_raw_test = data_PRI_raw_test[:, config.valid_PRI_idx]

# Feature augmentation
if config.augmentation:
    print('It will take some time to build the features of augmentation. If you do not want to wait, disable it!')
    # Sine augmentation
    data_DER_raw_sin2 = build_sin_2(data_DER_raw)
    data_DER_raw_test_sin2 = build_sin_2(data_DER_raw_test)

    data_PRI_raw_sin2 = build_sin_2(data_PRI_raw)
    data_PRI_raw_test_sin2 = build_sin_2(data_PRI_raw_test)

    # Power-series augmentation
    data_DER_raw_power = build_poly_power(data_DER_raw, degree=config.power_degree)
    data_DER_raw_test_power = build_poly_power(data_DER_raw_test, degree=config.power_degree)

    data_PRI_raw_power = build_poly_power(data_PRI_raw, degree=config.power_degree)
    data_PRI_raw_test_power = build_poly_power(data_PRI_raw_test, degree=config.power_degree)

    # Polynomial augmentation
    data_DER_raw = build_poly_with_interation_3(data_DER_raw)
    data_DER_raw_test = build_poly_with_interation_3(data_DER_raw_test)

    data_PRI_raw = build_poly_with_interation_3(data_PRI_raw)
    data_PRI_raw_test = build_poly_with_interation_3(data_PRI_raw_test)

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

print('Run Logistic Regression!')
function = method.logistic_regression

y = prediction
tx = data_DER

# Set hyper-parameters for the Gradient Descent
max_iters = config.max_iters
gamma = config.gamma
initial_w = np.zeros(tx.shape[-1])  # np.random.randn(tx.shape[-1])

# Create sub-dataset for cross validation
num_samples = len(y)
ratio = 0.1
seed = 42
train_sets, val_sets = cross_validation(num_samples, ratio=ratio, seed=seed)
record_acc_train = []
record_acc_val = []
record_w = []

acc_train, acc_val = [], []
for train_set, val_set in zip(train_sets, val_sets):
    y_train, y_val = y[train_set], y[val_set]
    tx_train, tx_val = tx[train_set], tx[val_set]
    loss, w = function(y_train, tx_train, initial_w, max_iters, gamma)
    prediction_train = method.sigmoid(tx_train.dot(w))
    prediction_train[prediction_train < 0.5] = -1
    prediction_train[prediction_train >= 0.5] = 1
    prediction_val = method.sigmoid(tx_val.dot(w))
    prediction_val[prediction_val < 0.5] = -1
    prediction_val[prediction_val >= 0.5] = 1

    acc_tr = (prediction_train == y_train).astype(float).sum() / len(y_train)
    acc_v = (prediction_val == y_val).astype(float).sum() / len(y_val)
    acc_train.append(acc_tr)
    acc_val.append(acc_v)
record_acc_train.append(sum(acc_train) / len(acc_train))
record_acc_val.append(sum(acc_val) / len(acc_val))

save_folder = 'results/logistic_regression' if not config.data_balance else 'results/logistic_regression_balance'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# if config.augmentation:
#     fig_save_path = os.path.join(save_folder, 'logistic_regression_degree_%d.png' % (config.power_degree))
# else:
#     fig_save_path = os.path.join(save_folder, 'logistic_regression.png')
# plot_train_test(record_acc_train, record_acc_val, fig_save_path)

_, w = function(y, tx, initial_w, max_iters, gamma)
prediction_test = method.sigmoid(data_DER_test.dot(w))
prediction_test[prediction_test < 0.5] = -1
prediction_test[prediction_test >= 0.5] = 1

if config.augmentation:
    csv_save_path = os.path.join(save_folder, 'logistic_regression_degree_%d.csv' % config.power_degree)
else:
    csv_save_path = os.path.join(save_folder, 'logistic_regression.csv')
save_data(prediction_test, csv_save_path)

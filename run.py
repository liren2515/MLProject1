
import os, sys
import time
import numpy as np
from lib.data import load_data, statistic_invalid, standardize, save_data, cross_validation, build_poly, add_bias
from lib import config
from lib import method
from lib.plots import plot_train_test

# You need to change the parameters in config.py.


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

data_DER_raw = data_DER_raw[:,config.valid_DER_idx]
data_PRI_raw = data_PRI_raw[:,config.valid_PRI_idx] # 8-D

data_DER_raw_test = data_DER_raw_test[:,config.valid_DER_idx]

# polynomial augmentation
if config.poly:
    data_DER_raw = build_poly(data_DER_raw, degree=config.poly_degree)
    data_DER_raw_test = build_poly(data_DER_raw_test, degree=config.poly_degree)

data_DER, mean_DER, std_DER = standardize(data_DER_raw)
data_DER_test = (data_DER_raw_test - mean_DER) / std_DER
data_DER = add_bias(data_DER)
data_DER_test = add_bias(data_DER_test)

function_name = config.methods[config.method_idx]

if function_name == 'GD':
    function = method.least_squares_GD

elif function_name == 'SGD':
    function = method.least_squares_SGD

elif function_name == 'LS':
    print('Least Square!')
    function = method.least_squares
    y = prediction
    tx = data_DER

    num_samples = len(y)
    ratio = 0.1
    seed = 42
    train_sets, val_sets = cross_validation(num_samples, ratio=ratio, seed=seed)
    acc_train, acc_val = [], []
    for train_set, val_set in zip(train_sets, val_sets):
        y_train, y_val = y[train_set], y[val_set]
        tx_train, tx_val = tx[train_set], tx[val_set]
        mse, w = function(y_train, tx_train)
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

    mse, w = function(y, tx)
    prediction_train = tx.dot(w)
    prediction_train[prediction_train<0] = -1
    prediction_train[prediction_train>=0] = 1

    print('Accuracy-train: ', sum(acc_train)/len(acc_train), ' Accuracy-val: ', sum(acc_val)/len(acc_val), ' Poly degree: ', config.poly_degree)

    save_folder = 'results/least_square' if not config.data_balance else 'results/least_square_balance'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    prediction_test = data_DER_test.dot(w)
    prediction_test[prediction_test<0] = -1
    prediction_test[prediction_test>=0] = 1

    if config.poly:
        csv_save_path = os.path.join(save_folder, 'least_square_degree_%d.csv'%config.poly_degree)
    else:
        csv_save_path = os.path.join(save_folder, 'least_square.csv')
    save_data(prediction_test, csv_save_path)


elif function_name == 'ridge':
    print('Ridge Regression!')
    function = method.ridge_regression
    y = prediction
    tx = data_DER
    lambdas = np.logspace(-5, -2, 30) #degree 1/2
    #lambdas = np.logspace(-6, -3, 15) #degree 3
    #lambdas = np.logspace(-15, -10, 30) #degree 7
    #lambdas = np.logspace(-16, -12, 30) #degree 10
    #lambdas = np.logspace(-16, -12, 30) #degree 15

    num_samples = len(y)
    ratio = 0.1
    seed = 42
    train_sets, val_sets = cross_validation(num_samples, ratio=ratio, seed=seed)
    record_acc_train = []   
    record_acc_val = []
    record_w = []
    for ind, lambda_ in enumerate(lambdas):
        acc_train, acc_val = [], []
        #tic = time.time()
        for train_set, val_set in zip(train_sets, val_sets):
            #tic_in = time.time()
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
            #toc_in = time.time()
            #print('Time-per-set: ', toc_in-tic_in)
        record_acc_train.append(sum(acc_train)/len(acc_train))
        record_acc_val.append(sum(acc_val)/len(acc_val))
        #toc = time.time()
        #print('Time=per-lambda: ', toc-tic)

    save_folder = 'results/ridge_regression' if not config.data_balance else 'results/ridge_regression_balance'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    if config.poly:
        fig_save_path = os.path.join(save_folder, 'ridge_regression_degree_%d.png'%config.poly_degree)
    else:
        fig_save_path = os.path.join(save_folder, 'ridge_regression.png')
    plot_train_test(record_acc_train, record_acc_val, lambdas, fig_save_path)

    idx_max = np.argmax(np.array(record_acc_val))
    lambda_ = lambdas[idx_max]
    _, w = function(y, tx, lambda_)
    prediction_test = data_DER_test.dot(w)
    prediction_test[prediction_test<0] = -1
    prediction_test[prediction_test>=0] = 1

    if config.poly:
        csv_save_path = os.path.join(save_folder, 'ridge_regression_degree_%d.csv'%config.poly_degree)
    else:
        csv_save_path = os.path.join(save_folder, 'ridge_regression.csv')
    save_data(prediction_test, csv_save_path)

elif function_name == 'logistic':
    function = method.logistic_regression

elif function_name == 'reg_logistic':
    function = method.reg_logistic_regression

else:
    print('Do Nothing!')



"""
invalid_DER = statistic_invalid(data_DER)
invalid_PRI = statistic_invalid(data_PRI)
#invalid_DER = {}
#invalid_PRI = {}
#for i in range()
print(data_PRI.shape, data_DER.shape, prediction.shape) # ((250000, 17), (250000, 13), (250000,))

for i in range(num_DER):
    print("DER feature: " +str(i) + ' invalid: ' +str(invalid_DER[i]))

for i in range(num_PRI):
    print("PRI feature: " +str(i) + ' invalid: ' +str(invalid_PRI[i]))
"""

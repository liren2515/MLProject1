First, you need to unzip the train/test.zip files in ./data (put the unziped files at ./data).

In lib/config.py, you can find the following hyper-parameters which you may want to change:
- augmentation: use feature augmentation;
- power_degree: the degree of power-series feature augmentation;
- data_balance: replicate samples to balance the data from 2 classes to have the same sample number;
- num_interval_lambda: the number of intervals you want to evaluate for the lambda;
- max_iters: the maximum number of iterations taken for GD/SGD;
- gamma: the step size for GD/SGD;

We exclude invalid variables whoes value can be -999.0, and consider polynomial/power-series/sine feature augmentation. The augmented features are normalized to have zero-means and unit-variances. We apply 10-fold cross validation. More details can be found in our report.

The test results will be saved in ./results automatically.

0. Best Result (which will give you 0.813/0.717 for the accurayc/F1 score):
run 'python run.py'.
(you can find the result at ./results/best_result)

1. Least Square Gradient Descent: 
run  'python run_least_square_GD.py'.

2. Least Square Stochastic Gradient Descent: 
run  'python run_least_square_SGD.py'.

3. Least Square: 
run  'python run_least_square'.

4. Ridge Regression: 
run  'python run_ridge_regression'.
(you can find the plot of accuracy-lambda in ./results/ ridge_regression.)

5. Logistic Regression: 
run  'python run_logistic_regression'.
(If you don't want to wait since it will take some time to build the augmented features, disable augmentation.)

6. Regularized Logistic Regression: 
run  'python run_reg_logistic_regression'.
(If you don't want to wait since it will take some time to build the augmented features, disable augmentation.)

In lib/config.py, you can find the following hyper-parameters which you may want to change:
- augmentation: use feature augmentation;
- power_degree: the degree of power-series feature augmentation;
- data_balance: replicate samples to balance the data from 2 classes to have the same sample number;
- num_interval_lambda: the number of intervals you want to evaluate for the lambda;
- max_iters: the maximum number of iterations taken for GD/SGD;
- gamma: the step size for GD/SGD;

The test results will be saved in ./results automatically.


1.

2.

3. Least Square: 
run  'python run_least_square'.

4. Ridge Regression: 
run  'python run_ridge_regression'.
(you can find the plot of accuracy-lambda in ./results/ ridge_regression.)

5. Regularized Logistic Regression: 
run  'python run_reg_logistic_regression'.
(If you don't want to wait since it will take some time to build the augmented features, disable augmentation.)

6. 
import numpy as np
import matplotlib.pyplot as plt

def plot_train_test(train_errors, val_errors, lambdas, save_path, degree=None):
    """
    train_errors, val_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = acc of a ridge regression on the train set
    * val_errors[0] = acc of the parameter found by ridge regression applied on the val set
    
    degree is just used for the title of the plot.
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, val_errors, color='r', marker='*', label="Val error")
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    if not type(degree) == type(None):
        plt.title("Ridge regression for polynomial degree " + str(degree))
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig(save_path)
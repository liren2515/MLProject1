import numpy as np

def load_data(path_dataset, sub_sample=False):
    """Load data and convert it to the metrics system."""
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[i for i in range(2,32)])
    data_PRI = data[:, 13:] # raw data
    data_DER = data[:, :13]
    prediction = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1],
        converters={0: lambda x: -1 if b"b" in x else 1}) # 'b': -1, 's': 1 to be decided

    # sub-sample
    if sub_sample:
        data_PRI = data_PRI[::50]
        data_DER = data_DER[::50]
        prediction = prediction[::50]

    return data_PRI, data_DER, prediction

def save_data(path_save, prediction, sample_data='data/sample-submission.csv', debug=False):
    """Save data to .csv format."""
    sample_data = np.genfromtxt(
        sample_data, delimiter=",", skip_header=1, usecols=[0, 1])

    if debug:
        prediction = np.random.randint(2, size=len(sample_data))*2-1
    sample_data[:, 1] = prediction
    sample_data = sample_data.astype(int)
    np.savetxt(path_save, sample_data, delimiter=',', fmt='%d', header='Id,Prediction', comments='')

    return

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def statistic_invalid(x):
    """Count the numbers of invalid features."""
    num = x.shape[-1]
    invalid = x == -999
    invalid = np.sum(invalid.astype(float), axis=0)
    return invalid.astype(int)
import numpy as np

def load_data(path_dataset, sub_sample=False, balance=False):
    """Load data and convert it to the metrics system."""
    data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[i for i in range(2,32)])
    data_PRI = data[:, 13:] # raw data
    data_DER = data[:, :13]
    #prediction = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[1], converters={0: lambda x: -1 if b"b" in x else 1}) # 'b': -1, 's': 1 to be decided
    prediction_str = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[1], dtype=str)
    prediction = np.ones(len(prediction_str))
    prediction[prediction_str=='b'] = -1

    # sub-sample
    if sub_sample:
        data_PRI = data_PRI[::50]
        data_DER = data_DER[::50]
        prediction = prediction[::50]

    #print((prediction==-1).astype(int).sum(), (prediction==1).astype(int).sum()) # 164333 85667
    if balance:
        num_miss = (prediction==-1).astype(int).sum() - (prediction==1).astype(int).sum()
        indx = np.where(prediction==1)[0]
        np.random.seed(42)
        np.random.shuffle(indx)
        indx = indx[:num_miss]
        data_DER_sample = data_DER[indx]
        data_PRI_sample = data_PRI[indx]
        prediction_sample = prediction[indx]
        data_DER = np.concatenate((data_DER, data_DER_sample), axis=0)
        data_PRI = np.concatenate((data_PRI, data_PRI_sample), axis=0)
        prediction = np.concatenate((prediction, prediction_sample), axis=0)

    return data_PRI, data_DER, prediction

def save_data(prediction, path_save, sample_data='data/sample-submission.csv', debug=False):
    """Save data to .csv format."""
    sample_data = np.genfromtxt(
        sample_data, delimiter=",", skip_header=1, usecols=[0, 1])

    if debug:
        prediction = np.random.randint(2, size=len(sample_data))*2-1
    sample_data[:, 1] = prediction
    sample_data = sample_data.astype(int)
    np.savetxt(path_save, sample_data, delimiter=',', fmt='%d', header='Id,Prediction', comments='')

    return

def split_data(x, y, ratio, seed=1):
    data = np.concatenate((x[:,np.newaxis], y[:,np.newaxis]), axis=-1)
    np.random.seed(seed)
    np.random.shuffle(data)
    num_train = int(len(x)*ratio)
    train_x = data[:num_train, 0].reshape(-1)
    train_y = data[:num_train, 1].reshape(-1)
    test_x = data[num_train:, 0].reshape(-1)
    test_y = data[num_train:, 1].reshape(-1)
    return train_x, train_y, test_x, test_y

def cross_validation(num_samples, ratio=0.1, seed=42):
    num_val_set = int(1/ratio)
    num_val_sample = int(num_samples*ratio)
    idx = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(idx)
    train_sets = []
    val_sets = []
    sample_set = set(idx.tolist())
    for i in range(num_val_set):
        val_set = idx[i*num_val_sample:(i+1)*num_val_sample].tolist()
        train_set = list(sample_set-set(val_set))
        val_sets.append(val_set)
        train_sets.append(train_set)
    print('%d train/val sets are created.'%num_val_set)
    return train_sets, val_sets

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

def build_power(x, degree):
    # Here we build power-series augmentation for N-D feature.
    poly = x.copy()
    for i in range(2, degree+1):
        poly = np.concatenate((poly, x**i), axis=-1)
    return poly

def build_poly_power(x, degree):
    # Here we build power-series augmentation starting from degree 4 for N-D feature.
    poly = x**4
    for i in range(5, degree+1):
        poly = np.concatenate((poly, x**i), axis=-1)
    return poly

def build_poly_with_interation(x):
    # Here we build polynomial augmentation of degree 2 for N-D feature.
    D = x.shape[-1]
    poly = x.copy()
    poly = np.concatenate((poly, x**2), axis=-1)
    for i in range(D):
        for j in range(i+1, D):
            x_ij = x[:,[i]]*x[:,[j]]
            #print(x_ij.shape, poly.shape)
            poly = np.concatenate((poly, x_ij), axis=-1)
    return poly

def build_poly_with_interation_3(x):
    # Here we build polynomial augmentation of degree 3 for N-D feature.
    D = x.shape[-1]
    poly = x.copy()
    poly = np.concatenate((poly, x**2, x**3), axis=-1)
    for i in range(D):
        for j in range(i+1, D):
            x_ij = x[:,[i]]*x[:,[j]]
            #print(x_ij.shape, poly.shape)
            poly = np.concatenate((poly, x_ij), axis=-1)
            poly = np.concatenate((poly, x_ij*x[:,[i]]), axis=-1)
            for k in range(j, D):
                poly = np.concatenate((poly, x_ij*x[:,[k]]), axis=-1)
            
    return poly

def build_sin_2(x):
    # Here we build sine augmentation of degree 2 for N-D feature.
    D = x.shape[-1]
    sin = np.concatenate((np.sin(x), np.cos(x), np.sin(2*x), np.cos(2*x)), axis=-1)
    for i in range(D):
        sin_i = np.sin(x[:,[i]])
        cos_i = np.cos(x[:,[i]])
        for j in range(i+1, D):
            sin_j = np.sin(x[:,[j]])
            cos_j = np.cos(x[:,[j]])
            sin = np.concatenate((sin, sin_i*sin_j, sin_i*cos_j, cos_i*sin_j, cos_i*cos_j), axis=-1)
    return sin

def add_bias(x):
    N = len(x)
    x = np.concatenate((np.ones((N,1)), x), axis=-1)
    return x
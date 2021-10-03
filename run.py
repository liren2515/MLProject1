from lib.data import load_data, statistic_invalid, standardize, save_data
from lib import config

path_dataset = 'data/train.csv'
data_PRI_raw, data_DER_raw, prediction = load_data(path_dataset)

num_sample = len(prediction)
num_DER = data_DER_raw.shape[-1]
num_PRI = data_PRI_raw.shape[-1]

data_DER_raw = data_DER_raw[config.valid_DER_idx]
data_PRI_raw = data_PRI_raw[config.valid_PRI_idx]

data_DER, mean_DER, std_DER = standardize(data_DER_raw)

path_save = 'results/tmp.csv'
save_data(path_save, None, debug=True)





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

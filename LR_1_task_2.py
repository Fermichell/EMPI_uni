import numpy as np
from sklearn import preprocessing


flat_values = [1.3, 3.9, 6.2, 4.9, 2.2, -4.3, -2.2, 6.5, 4.1, -5.2, -3.4, -5.2]
threshold = 2.0
input_data = np.array(flat_values, dtype=float).reshape(4, 3)

print("Input data:\n", input_data)

data_binarized = preprocessing.Binarizer(threshold=threshold).transform(input_data)
print(f"\nBinarized data (threshold={threshold}):\n", data_binarized)

print("\nBEFORE:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)
print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = scaler_minmax.fit_transform(input_data)
print("\nMin–Max scaled data:\n", data_scaled_minmax)

data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL1 normalized data:\n", data_normalized_l1)
print("\nL2 normalized data:\n", data_normalized_l2)

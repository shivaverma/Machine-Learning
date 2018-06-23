import numpy as np

X = np.random.randint(0, 5, (10, 2))


mean_col = X.mean(axis=0)
std_col = X.std(axis=0)

# Mean normalization
X_norm = (X - mean_col)/std_col

avg = np.mean(X_norm)
print("Average of X_norm: " + str(avg))

avg_min = np.mean(np.min(X_norm, axis=0))
print("Average of min: " + str(avg_min))

avg_min = np.mean(np.max(X_norm, axis=0))
print("Average of max: " + str(avg_min))

rows, _ = X_norm.shape
row_indices = np.random.permutation(rows)

X_train = X_norm[row_indices[:6]]
X_val = X_norm[row_indices[6:8]]
X_test = X_norm[row_indices[8:10]]

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

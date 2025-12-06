import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.metrics import accuracy_score

df = pd.read_csv("winequality-white.csv", sep=';')
def multiclass_label(q):
    if q <= 5:
        return 0
    elif q in [6, 7]:
        return 1
    else:
        return 2

df["multiclass"] = df["quality"].apply(multiclass_label)

X = df.drop(["quality", "multiclass"], axis=1)
y = df["multiclass"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X_scaled)

# 90% training, 10% validation after dimension reduction
X_train_pca, X_val_pca, y_train, y_val = train_test_split(
    X_reduced, y,
    test_size=0.10,
    random_state=42,
    stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#KNN
knn_bin = KNeighborsClassifier(n_neighbors=1)

start_train = time.time()
knn_bin.fit(X_train_pca, y_train)
end_train = time.time()

y_train_pred_knn = knn_bin.predict(X_train_pca)
train_acc_knn = accuracy_score(y_train, y_train_pred_knn)
train_time_knn = end_train - start_train

print("\nKNN")
print(f"Train accuracy: {train_acc_knn:.4f}")
print(f"Train runtime: {train_time_knn:.6f} seconds")

start_test = time.time()
y_val_pred_knn = knn_bin.predict(X_val_pca)
end_test = time.time()

val_acc_knn = accuracy_score(y_val, y_val_pred_knn)
test_time_knn = end_test - start_test

print("\nKNN — Validation Results")
print(f"Validation accuracy: {val_acc_knn:.4f}")
print(f"Validation (testing) runtime: {test_time_knn:.6f} seconds")

#Decision Tree
dt_bin = DecisionTreeClassifier(random_state=42, max_depth=20)

start_train = time.time()
dt_bin.fit(X_train_pca, y_train)
end_train = time.time()

y_train_pred_dt = dt_bin.predict(X_train_pca)
train_acc_dt = accuracy_score(y_train, y_train_pred_dt)
train_time_dt = end_train - start_train

print("\nDecision Tree")
print(f"Train accuracy: {train_acc_dt:.4f}")
print(f"Train runtime: {train_time_dt:.6f} seconds")

start_test = time.time()
y_val_pred_dt = dt_bin.predict(X_val_pca)
end_test = time.time()

val_acc_dt = accuracy_score(y_val, y_val_pred_dt)
test_time_dt = end_test - start_test

print("\nDecision Tree — Validation Results")
print(f"Validation accuracy: {val_acc_dt:.4f}")
print(f"Validation (testing) runtime: {test_time_dt:.6f} seconds")
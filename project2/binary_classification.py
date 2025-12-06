import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("winequality-white.csv", sep=';')

df["binary"] = (df["quality"] >= 7).astype(int)

X = df.drop(["quality", "binary"], axis=1)
y= df["binary"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.10,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Validation size (for final test):", X_val.shape[0])

sc_bin = StandardScaler()
X_train_bin_sc = sc_bin.fit_transform(X_train)
X_val_bin_sc = sc_bin.transform(X_val)


knn_bin = KNeighborsClassifier(n_neighbors=1)

start_train = time.time()
knn_bin.fit(X_train_bin_sc, y_train)
end_train = time.time()

y_train_pred_knn = knn_bin.predict(X_train_bin_sc)
train_acc_knn = accuracy_score(y_train, y_train_pred_knn)
train_time_knn = end_train - start_train

start_test = time.time()
y_final_pred_knn = knn_bin.predict(X_val_bin_sc)
end_test = time.time()

final_acc_knn = accuracy_score(y_val, y_final_pred_knn)
test_time_knn = end_test - start_test

print("\nKNN (binary)")
print(f"Train accuracy: {train_acc_knn:.4f}")
print(f"Train runtime: {train_time_knn:.6f} seconds")
print(f"Final validation accuracy: {final_acc_knn:.4f}")
print(f"Final validation (testing) runtime: {test_time_knn:.6f} seconds")


dt_bin = DecisionTreeClassifier(random_state=42, max_depth=20)

start_train = time.time()
dt_bin.fit(X_train, y_train)
end_train = time.time()

y_train_pred_dt = dt_bin.predict(X_train)
train_acc_dt = accuracy_score(y_train, y_train_pred_dt)
train_time_dt = end_train - start_train

start_test = time.time()
y_final_pred_dt = dt_bin.predict(X_val)
end_test = time.time()

final_acc_dt = accuracy_score(y_val, y_final_pred_dt)
test_time_dt = end_test - start_test

print("\nDecision Tree (binary)")
print(f"Train accuracy: {train_acc_dt:.4f}")
print(f"Train runtime: {train_time_dt:.6f} seconds")
print(f"Final validation accuracy: {final_acc_dt:.4f}")
print(f"Final validation (testing) runtime: {test_time_dt:.6f} seconds")

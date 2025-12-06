import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.10,
    random_state=42,
    stratify=y
)

print("Train size (multiclass):", X_train.shape[0])
print("Validation size (for final test, multiclass):", X_val.shape[0])

sc_mc = StandardScaler()
X_train_mc_sc = sc_mc.fit_transform(X_train)
X_val_mc_sc = sc_mc.transform(X_val)

#KNN
knn_mc = KNeighborsClassifier(n_neighbors=1)
start_train = time.time()
knn_mc.fit(X_train_mc_sc, y_train)
end_train = time.time()

y_train_pred_knn = knn_mc.predict(X_train_mc_sc)
train_acc_knn = accuracy_score(y_train, y_train_pred_knn)
train_time_knn = end_train - start_train

#test
start_test = time.time()
y_val_pred_knn = knn_mc.predict(X_val_mc_sc)
end_test = time.time()

val_acc_knn = accuracy_score(y_val, y_val_pred_knn)
test_time_knn = end_test - start_test

print("\nKNN (multiclass)")
print(f"Train accuracy: {train_acc_knn:.4f}")
print(f"Train runtime: {train_time_knn:.6f} seconds")
print(f"Validation accuracy: {val_acc_knn:.4f}")
print(f"Validation (testing) runtime: {test_time_knn:.6f} seconds")

#Decision Tree
dt_mc = DecisionTreeClassifier(random_state=42, max_depth=12)

start_train = time.time()
dt_mc.fit(X_train, y_train)
end_train = time.time()

y_train_pred_dt = dt_mc.predict(X_train)
train_acc_dt = accuracy_score(y_train, y_train_pred_dt)
train_time_dt = end_train - start_train

#test
start_test = time.time()
y_val_pred_dt = dt_mc.predict(X_val)
end_test = time.time()

val_acc_dt = accuracy_score(y_val, y_val_pred_dt)
test_time_dt = end_test - start_test

print("\nDecision Tree (multiclass)")
print(f"Train accuracy: {train_acc_dt:.4f}")
print(f"Train runtime: {train_time_dt:.6f} seconds")
print(f"Validation accuracy: {val_acc_dt:.4f}")
print(f"Validation (testing) runtime: {test_time_dt:.6f} seconds")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("winequality-white.csv", sep=';')
# Binary label
df["binary"] = (df["quality"] >= 7).astype(int)

# Multiclass label
def multiclass_label(q):
    if q <= 5:
        return 0
    if q in [6, 7]:
        return 1
    return 2
df["multiclass"] = df["quality"].apply(multiclass_label)

# Features
X = df.drop(["quality", "binary", "multiclass"], axis=1)
y_bin = df["binary"]
y_mc = df["multiclass"]

# Split dataset: 80% Train, 10% Validation, 10% Final Validation
# First split: Train (80%) + Temp (20%)
X_train_bin, X_temp_bin, y_train_bin, y_temp_bin = train_test_split(
    X, y_bin, test_size=0.20, random_state=42, stratify=y_bin
)

X_train_mc, X_temp_mc, y_train_mc, y_temp_mc = train_test_split(
    X, y_mc, test_size=0.20, random_state=42, stratify=y_mc
)

# Second split Temp (20%) into Validation (10%) + Final Validation (10%)
X_val_bin, X_final_bin, y_val_bin, y_final_bin = train_test_split(
    X_temp_bin, y_temp_bin, test_size=0.50, random_state=42, stratify=y_temp_bin
)

X_val_mc, X_final_mc, y_val_mc, y_final_mc = train_test_split(
    X_temp_mc, y_temp_mc, test_size=0.50, random_state=42, stratify=y_temp_mc
)

# Standardize for KNN (fit ONLY on training data)
sc_bin = StandardScaler()
X_train_bin_sc = sc_bin.fit_transform(X_train_bin)
X_val_bin_sc = sc_bin.transform(X_val_bin)
X_final_bin_sc = sc_bin.transform(X_final_bin)

sc_mc = StandardScaler()
X_train_mc_sc = sc_mc.fit_transform(X_train_mc)
X_val_mc_sc = sc_mc.transform(X_val_mc)
X_final_mc_sc = sc_mc.transform(X_final_mc)


# KNN
knn_bin = KNeighborsClassifier(n_neighbors=5)
knn_bin.fit(X_train_bin_sc, y_train_bin)

knn_mc = KNeighborsClassifier(n_neighbors=5)
knn_mc.fit(X_train_mc_sc, y_train_mc)

# Decision Tree
dt_bin = DecisionTreeClassifier(random_state=42)
dt_bin.fit(X_train_bin, y_train_bin)

dt_mc = DecisionTreeClassifier(random_state=42)
dt_mc.fit(X_train_mc, y_train_mc)


# Evaluate on FINAL VALIDATION SET ONLY
print("KNN Binary Accuracy:", accuracy_score(y_final_bin, knn_bin.predict(X_final_bin_sc)))
print("KNN Multiclass Accuracy:", accuracy_score(y_final_mc, knn_mc.predict(X_final_mc_sc)))

print("Decision Tree Binary Accuracy:", accuracy_score(y_final_bin, dt_bin.predict(X_final_bin)))
print("Decision Tree Multiclass Accuracy:", accuracy_score(y_final_mc, dt_mc.predict(X_final_mc)))

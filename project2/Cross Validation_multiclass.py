import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("winequality-white.csv", sep=';')
def multiclass_label(q):
    if q <= 5:
        return 0       # low quality
    elif q in [6, 7]:
        return 1       # medium quality
    else:
        return 2       # high quality

df["multiclass"] = df["quality"].apply(multiclass_label)

X = df.drop(["quality", "multiclass"], axis=1)
y = df["multiclass"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.10,
    random_state=42,
    stratify=y
)
print("Multiclass training size:", X_train.shape[0])
print("Multiclass validation size:", X_val.shape[0])

# 5-fold CV setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# KNN cross-validation visualization (Multiclass)
k_values = [1, 3, 5, 7, 9, 11, 13]
knn_mean_scores = []
knn_std_scores = []

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

for k in k_values:
    knn_pipeline.set_params(knn__n_neighbors=k)
    scores = cross_val_score(
        knn_pipeline,
        X_train, y_train,
        cv=cv,
        scoring="accuracy"
    )
    knn_mean_scores.append(scores.mean())
    knn_std_scores.append(scores.std())

plt.figure(figsize=(7,5))
plt.errorbar(k_values, knn_mean_scores, yerr=knn_std_scores, marker="o", capsize=5)
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("5-fold CV Accuracy")
plt.title("KNN: 5-fold Cross-Validation Accuracy vs k (Multiclass Classification)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()



# Decision Tree cross-validation visualization (Multiclass)
depth_values = [10, 12, 14, 16, 18, 20, 22]
dt_mean_scores = []
dt_std_scores = []

for d in depth_values:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    scores = cross_val_score(
        dt,
        X_train, y_train,
        cv=cv,
        scoring="accuracy"
    )
    dt_mean_scores.append(scores.mean())
    dt_std_scores.append(scores.std())

plt.figure(figsize=(7,5))
plt.errorbar(depth_values, dt_mean_scores, yerr=dt_std_scores, marker="o", capsize=5)
plt.xlabel("Max Depth")
plt.ylabel("5-fold CV Accuracy")
plt.title("Decision Tree: 5-fold Cross-Validation Accuracy vs max_depth (Multiclass Classification)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

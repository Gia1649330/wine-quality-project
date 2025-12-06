import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("winequality-white.csv", sep=';')

df["binary"] = (df["quality"] >= 7).astype(int)

X = df.drop(["quality", "binary"], axis=1)
y = df["binary"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.10,
    random_state=42,
    stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

plt.figure()
plt.errorbar(k_values, knn_mean_scores, yerr=knn_std_scores, marker="o")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("5-fold CV Accuracy")
plt.title("KNN: 5-fold Cross-Validation Accuracy vs k (Binary Classification)")
plt.tight_layout()
plt.show()


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

plot_depth_values = [d if d is not None else 0 for d in depth_values]
plt.figure()
plt.errorbar(plot_depth_values, dt_mean_scores, yerr=dt_std_scores, marker="o")
plt.xlabel("Max Depth")
plt.ylabel("5-fold CV Accuracy")
plt.title("Decision Tree: 5-fold Cross-Validation Accuracy vs max_depth (Binary Classification)")
plt.tight_layout()
plt.show()
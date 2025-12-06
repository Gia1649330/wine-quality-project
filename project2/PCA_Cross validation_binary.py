import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("winequality-white.csv", sep=';')

df["binary"] = (df["quality"] >= 7).astype(int)
X = df.drop(["quality", "binary"], axis=1)
y= df["binary"]

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
print("Training size (PCA):", X_train_pca.shape[0])
print("Validation size (PCA):", X_val_pca.shape[0])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

k_values = [1, 3, 5, 7, 9, 11, 13]
knn_mean_scores = []
knn_std_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(
        knn,
        X_train_pca, y_train,
        cv=cv,
        scoring="accuracy"
    )
    knn_mean_scores.append(scores.mean())
    knn_std_scores.append(scores.std())
    print(f"KNN (PCA): k={k}, mean CV accuracy = {scores.mean():.4f}")

plt.figure(figsize=(7,5))
plt.errorbar(k_values, knn_mean_scores, yerr=knn_std_scores,
             marker="o", capsize=5)
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("5-fold CV Accuracy")
plt.title("KNN with PCA: 5-fold CV Accuracy vs k (Binary Classification)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

depth_values = [10, 12, 14, 16, 18, 20, 22]
dt_mean_scores = []
dt_std_scores = []

for d in depth_values:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    scores = cross_val_score(
        dt,
        X_train_pca, y_train,
        cv=cv,
        scoring="accuracy"
    )
    dt_mean_scores.append(scores.mean())
    dt_std_scores.append(scores.std())
    print(f"Decision Tree (PCA): max_depth={d}, mean CV accuracy = {scores.mean():.4f}")

plt.figure(figsize=(7,5))
plt.errorbar(depth_values, dt_mean_scores, yerr=dt_std_scores,
             marker="o", capsize=5)
plt.xlabel("Max Depth")
plt.ylabel("5-fold CV Accuracy")
plt.title("Decision Tree with PCA: 5-fold CV Accuracy vs max_depth (Binary Classification)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
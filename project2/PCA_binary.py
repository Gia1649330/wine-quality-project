import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("winequality-white.csv", sep=';')

df["binary"] = (df["quality"] >= 7).astype(int)
X = df.drop(["quality", "binary"], axis=1)
y= df["binary"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X_scaled)

X_train_pca, X_val_pca, y_train, y_val = train_test_split(
    X_reduced, y,
    test_size=0.10,
    random_state=42,
    stratify=y
)

print("Training size (PCA, binary):", X_train_pca.shape[0])
print("Validation size (PCA, binary):", X_val_pca.shape[0])

pc_cols = [f"PC{i}" for i in range(1, X_train_pca.shape[1] + 1)]

train_pca_df = pd.DataFrame(X_train_pca, columns=pc_cols)
train_pca_df["binary"] = y_train.values

val_pca_df = pd.DataFrame(X_val_pca, columns=pc_cols)
val_pca_df["binary"] = y_val.values

train_pca_df.to_csv("train_binary_pca.csv", index=False)
val_pca_df.to_csv("validation_binary_pca.csv", index=False)

print("\nSaved files:")
print(" - train_binary_pca.csv")
print(" - validation_binary_pca.csv")
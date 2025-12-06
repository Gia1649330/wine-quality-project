import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("winequality-white.csv", sep=';')
def multiclass_label(q):
    if q <= 5:
        return 0
    elif q in [6, 7]:
        return 1
    else:
        return 2

df["multiclass"] = df["quality"].apply(multiclass_label)

feature_cols = [c for c in df.columns if c not in ["quality", "multiclass"]]
X = df[feature_cols]
y = df["multiclass"]

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

print("Multiclass PCA training size:", X_train_pca.shape[0])
print("Multiclass PCA validation size:", X_val_pca.shape[0])

pc_cols = [f"PC{i}" for i in range(1, X_train_pca.shape[1] + 1)]

# Create DataFrames and put response variable last
train_df = pd.DataFrame(X_train_pca, columns=pc_cols)
train_df["multiclass"] = y_train.values

val_df = pd.DataFrame(X_val_pca, columns=pc_cols)
val_df["multiclass"] = y_val.values

# Save to CSV
train_df.to_csv("train_multiclass_pca.csv", index=False)
val_df.to_csv("validation_multiclass_pca.csv", index=False)

print("\nSaved files:")
print(" - train_multiclass_pca.csv")
print(" - validation_multiclass_pca.csv")
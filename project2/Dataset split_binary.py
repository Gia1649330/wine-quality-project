from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("winequality-white.csv", sep=';')

df["binary"] = (df["quality"] >= 7).astype(int)

# Features + label (drop quality but KEEP binary as last column)
df_binary = df.drop(columns=["quality"])

# Split into training (90%) and validation (10%)
train_df, val_df = train_test_split(
    df_binary,
    test_size=0.10,
    random_state=42,
    stratify=df_binary["binary"]
)

print("Training size:", train_df.shape)
print("Validation size:", val_df.shape)

# Save as CSV files
train_df.to_csv("train_binary.csv", index=False)
val_df.to_csv("validation_binary.csv", index=False)

print("\nSaved files:")
print(" - train_binary.csv")
print(" - validation_binary.csv")

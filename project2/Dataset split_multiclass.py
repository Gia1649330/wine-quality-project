import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("winequality-white.csv", sep=';')

def multiclass_label(q):
    if q <= 5:
        return 0       # low quality
    elif q in [6, 7]:
        return 1       # medium quality
    else:
        return 2       # high quality

df["multiclass"] = df["quality"].apply(multiclass_label)

# Final dataset (drop only 'quality')
df_mc = df.drop(columns=["quality"])

# Split into training (90%) and validation (10%)
train_mc, val_mc = train_test_split(
    df_mc,
    test_size=0.10,
    random_state=42,
    stratify=df_mc["multiclass"]
)

print("Multiclass training size:", train_mc.shape)
print("Multiclass validation size:", val_mc.shape)

# Save to CSV (last column = multiclass)
train_mc.to_csv("train_multiclass.csv", index=False)
val_mc.to_csv("validation_multiclass.csv", index=False)

print("\nSaved files:")
print(" - train_multiclass.csv")
print(" - validation_multiclass.csv")

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("winequality-white.csv", sep=';')
num_cols = [c for c in df.columns if c != 'quality']

for col in num_cols:
    plt.figure()
    plt.hist(df[col], bins=30)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

plt.figure()
df['quality'].value_counts().sort_index().plot(kind='bar')
plt.title("Bar plot of quality")
plt.xlabel("Quality score")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

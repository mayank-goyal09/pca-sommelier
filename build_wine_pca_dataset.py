import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load UCI Wine dataset via sklearn
wine = load_wine(as_frame=True)
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

df = X.copy()
df["target"] = y
df["target_name"] = df["target"].map(dict(enumerate(target_names)))

# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA (keep, say, 5 components for richer CSV)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
pca_df = pd.DataFrame(X_pca, columns=pca_cols)

# 4. Combine original + PCA
full_df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)

# 5. Optionally, replicate rows a bit to make it “bigger”
# (purely for a feel of a big CSV; real projects would use a bigger source)
full_df_big = pd.concat([full_df] * 5, ignore_index=True)

# 6. Save to CSV
full_df_big.to_csv("wine_pca_dataset.csv", index=False)

print("Saved wine_pca_dataset.csv with shape:", full_df_big.shape)

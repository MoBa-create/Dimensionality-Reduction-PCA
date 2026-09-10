from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

np.random.seed(42)
n_samples = 1000

age = np.random.randint(18, 70, n_samples)
income = age * 1.8 + np.random.normal(20, 5, n_samples)
spending = 100 - (age * 0.9) + np.random.normal(0, 10, n_samples)
credit = income * 8 + np.random.normal(100, 30, n_samples)
balance = income * 1200 + np.random.normal(2000, 500, n_samples)

df = pd.DataFrame({
    "Age": age,
    "Income": income.round(2),
    "SpendingScore": spending.round(2),
    "CreditScore": credit.round(2),
    "Balance": balance.round(2)
})

print("=== Original Data Shape ===")
print(df.shape)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

var_ratio = pca.explained_variance_ratio_

print("\n=== PCA Results ===")
print(f"Explained Variance per Component: {var_ratio.round(4)}")
print(f"Total Preserved Information: {sum(var_ratio) * 100:.2f}%")

df_pca = pd.DataFrame(x_pca, columns=["PC1", "PC2"])
print("\n=== Reduced Data (First 5 rows) ===")
print(df_pca.head())

joblib.dump(pca, os.path.join(OUTPUTS_DIR, "pca_model.pkl"))
joblib.dump(scaler, os.path.join(OUTPUTS_DIR, "scaler.pkl"))
df_pca.to_csv(os.path.join(OUTPUTS_DIR, "pca_transformed_data.csv"), index=False)

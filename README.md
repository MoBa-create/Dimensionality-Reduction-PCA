# Dimensionality Reduction Pipeline using PCA (Unsupervised Learning)

An end-to-end Machine Learning pipeline built with **Scikit-Learn** and **Principal Component Analysis (PCA)** to reduce feature dimensionality while preserving maximum dataset variance.

---

## 📌 Key Highlights
* **Correlated Feature Reduction:** Compressed 5 financial & demographic features (`Age`, `Income`, `SpendingScore`, `CreditScore`, `Balance`) into 2 Principal Components ($PC_1, PC_2$).
* **Data Scaling:** Applied `StandardScaler` to ensure scale-invariance before applying PCA.
* **Variance Preservation:** Achieved a **99.26% Total Preserved Variance Ratio** (PC1: 93.12%, PC2: 6.14%), minimizing information loss.
* **Pipeline Serialization:** Serialized fitted `PCA` and `StandardScaler` objects using `joblib`.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Scikit-Learn, Pandas, NumPy, Joblib

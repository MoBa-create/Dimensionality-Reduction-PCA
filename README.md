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

---

## 📁 Repository Structure
```text
Dimensionality-Reduction-PCA/
│── pca_pipeline.py           # Main script for data scaling, PCA, and serialization
│── README.md                 # Project documentation
│── requirements.txt          # Python dependencies
│── .gitignore                # Git ignore configuration
└── outputs/                  # Saved artifacts
    ├── pca_model.pkl         # Fitted PCA model instance
    ├── scaler.pkl            # Fitted StandardScaler instance
    └── pca_transformed_data.csv # Reduced 2D dataset (PC1, PC2)

🚀 How to Run

1 . Clone the repository:

	git clone https://github.com/MoBa-create/Dimensionality-Reduction-PCA.git
cd Dimensionality-Reduction-PCA

2 . Install dependencies:

	pip install -r requirements.txt

3 . Execute the PCA pipeline:

	python pca_pipeline.py
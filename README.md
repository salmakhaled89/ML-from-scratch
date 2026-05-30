# ML-from-scratch
Custom Python implementation of PCA, LDA, and SVM from scratch, evaluated and benchmarked against Scikit-Learn on an anonymous credit card fraud detection dataset.
# Machine Learning Core Architectures From Scratch

A mathematical and programmatic investigation into core Machine Learning algorithms. This repository contains custom, raw Python implementations of Dimensionality Reduction (**PCA**, **LDA**) and Classification (**SVM**) techniques built strictly using **NumPy**, **Pandas**, and **SciPy**, bench-marked directly against **Scikit-Learn** production models.

The models are evaluated on a highly imbalanced, real-world Financial Fraud Detection dataset containing anonymous numerical features ($V_1$ to $V_{10}$) mapped to binary classes (`0: Non-Fraud`, `1: Fraud`).

##  Tech Stack & Mathematical Concepts
- **Language:** Python 3
- **Core Vectorization:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Evaluations:** Scikit-Learn (Used exclusively for benchmarking and metrics analysis)

---

##  Algorithmic Frameworks Built From Scratch

### 1. Principal Component Analysis (PCA)
- **Objective:** Unsupervised linear dimensionality reduction.
- **Mathematical Approach:** Built via the **Covariance Matrix** and **Eigenvalue Decomposition (EVD)**. Maximizes variance by projecting data onto orthogonal Principal Components ($PC_1, PC_2, \dots$).
- **Verification:** Benchmarked against `sklearn.decomposition.PCA` by checking total explained variance ratios and reconstructing singular values.

### 2. Linear Discriminant Analysis (LDA)
- **Objective:** Supervised dimensionality reduction and linear classification.
- **Mathematical Approach:** Maximizes class separability by maximizing the ratio of between-class variance ($S_B$) to within-class variance ($S_W$) using Fisher's Linear Discriminant optimization.

### 3. Support Vector Machine (SVM)
- **Objective:** Maximal-margin binary classification.
- **Mathematical Approach:** Formulated using the **Dual Optimization Problem** solved via Quadratic Programming. Implements custom hyper-plane margins to handle lineally separating boundaries.

---

##  Dataset & Benchmarking Insights
- **Features:** 10 transformed, continuous anonymized features ($V_1$ through $V_{10}$).
- **Target:** Class binary attribute (`0` vs `1`).
- **Benchmarking Protocol:** Both custom scratch models and Scikit-Learn counterparts were evaluated using identical training/testing splits ($80\% - 20\%$).
- **Key Discovery:** Custom NumPy array vectorization achieved up to $99\%$ alignment in prediction mapping compared to Scikit-Learn, verifying mathematical convergence.
-

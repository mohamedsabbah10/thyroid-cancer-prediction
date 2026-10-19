# Thyroid Cancer Prediction Using Machine Learning

## Project Overview

This project explores the use of machine learning classification algorithms to predict the `Diagnosis` label in a thyroid cancer dataset. It compares multiple models, evaluates their performance using classification metrics and cross-validation, and investigates potential target leakage associated with the `Thyroid_Cancer_Risk` feature.

The project demonstrates an end-to-end machine learning workflow, from exploratory data analysis and preprocessing to model evaluation and interpretation.

> **Disclaimer:** This is an educational machine learning project, not a clinically validated diagnostic system. Its predictions must not be used to diagnose thyroid cancer or guide medical decisions.

## Objectives

- Explore and understand the dataset through exploratory data analysis (EDA).
- Examine the distribution of benign and malignant diagnoses.
- Prepare numerical and categorical features for machine learning.
- Train and compare multiple classification algorithms.
- Evaluate model performance using several classification metrics.
- Apply stratified cross-validation to assess model stability.
- Investigate potential target leakage involving `Thyroid_Cancer_Risk`.
- Document the limitations of using dataset-based predictions in a medical context.

## Technologies and Libraries

- **Python** — Programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computing
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical data visualization
- **Scikit-learn** — Preprocessing, model training, validation, and evaluation
- **Jupyter Notebook** — Interactive development and documentation

## Machine Learning Models

The notebook compares three classification algorithms:

| Model | Purpose |
|---|---|
| Logistic Regression | Establishes an interpretable classification baseline. |
| Decision Tree | Captures nonlinear relationships using decision rules. |
| Random Forest | Combines multiple decision trees to model complex patterns. |

The models are evaluated on the same stratified train-test split to support a consistent comparison.

## Project Workflow

### 1. Data Loading and Inspection

The dataset is loaded from a CSV file. Initial inspection includes:

- Dataset dimensions and column information
- Missing-value counts
- Duplicate-row and duplicate-patient identifier checks
- Descriptive statistics for numerical and categorical columns

### 2. Exploratory Data Analysis

The notebook examines the dataset through visualizations and summary tables, including:

- Diagnosis class distribution
- Age distribution
- Thyroid nodule size distribution
- Diagnosis proportions across `Thyroid_Cancer_Risk` categories

These analyses help identify patterns in the available data before model training.

### 3. Data Preparation

The target variable is `Diagnosis`, encoded as:

- `Benign` → `0`
- `Malignant` → `1`

The identifier columns `Unnamed: 0` and `Patient_ID` are excluded from the predictive features.

The dataset is divided into training and testing subsets using an 80/20 split, with stratification to preserve the target class proportions.

### 4. Preprocessing

A Scikit-learn `ColumnTransformer` and `Pipeline` are used to prepare the features:

- **Numerical features:** Standardized using `StandardScaler`.
- **Categorical features:** Encoded using `OneHotEncoder` with `handle_unknown="ignore"`.

Preprocessing is fitted on the training data within each pipeline, helping prevent information from the test set from influencing the fitted transformations.

### 5. Model Training and Evaluation

Each model is trained on the training subset and evaluated on the held-out test subset using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted positive cases that are actually positive.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.

The notebook presents the results in a comparison table and a performance visualization. The model with the highest test-set F1 Score is selected for further inspection.

### 6. Confusion Matrix and Classification Report

The selected model is examined using a confusion matrix and a classification report.

These outputs provide additional information about classification errors and the model's precision, recall, and F1 Score for each class.

### 7. Cross-Validation

Stratified 5-fold cross-validation is performed on the training subset using F1 Score as the evaluation metric.

The notebook reports the mean and standard deviation of the cross-validation scores for each model, providing additional insight into performance across different training and validation folds.

### 8. Potential Target Leakage Investigation

The feature `Thyroid_Cancer_Risk` is investigated because it may contain information related to the target diagnosis.

Two scenarios are compared:

- **Scenario A:** Models trained with `Thyroid_Cancer_Risk` included.
- **Scenario B:** Models trained with `Thyroid_Cancer_Risk` excluded.

The resulting F1 Scores and their differences are compared to assess how strongly model performance depends on this feature.

A performance difference alone does not prove target leakage. The feature's origin and the timing of its availability must be investigated to determine whether it contains information that would not legitimately be available at prediction time.

## Repository Structure

The repository is intended to follow this structure:

```text
thyroid-cancer-prediction/
├── Thyroid_Cancer_Prediction_Improved.ipynb
├── data/
│   └── Thyroid Cancer Dataset.csv
├── README.md
└── .gitignore
```

The CSV filename and notebook path should match the actual files in the repository. The dataset is not included automatically and must be obtained from a source that permits its use and redistribution.

## Getting Started

### Prerequisites

Install Python and Jupyter Notebook, or use an environment such as JupyterLab or Google Colab.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/thyroid-cancer-prediction.git
cd thyroid-cancer-prediction
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Prepare the Dataset

Place the dataset at:

```text
data/Thyroid Cancer Dataset.csv
```

Ensure the notebook loads it using a repository-relative path:

```python
DATA_PATH = "data/Thyroid Cancer Dataset.csv"
data = pd.read_csv(DATA_PATH)
```

Update the path if your directory structure differs.

### 4. Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `Thyroid_Cancer_Prediction_Improved.ipynb` and run the cells sequentially, from data loading through the final conclusions.

## Reproducibility

The notebook uses a fixed random seed:

```python
RANDOM_STATE = 42
```

It also uses a stratified train-test split and shuffled stratified 5-fold cross-validation with a fixed random seed.

These settings support reproducibility under the same data, software environment, and configuration. They do not guarantee identical results across every software version or execution environment.

## Results

Model performance should be reported only after successfully running the notebook on the intended dataset.

The final results section should document:

- The best model according to test-set F1 Score
- The best model according to mean cross-validation F1 Score
- The observed accuracy, precision, recall, and F1 Score
- The effect of excluding `Thyroid_Cancer_Risk`
- The limitations and interpretation of the observed results

No performance figures are reported here because they have not been established from an executed notebook run.

## Limitations

- The results depend on the quality, representativeness, and reliability of the dataset.
- A single train-test split may not fully represent performance on unseen populations.
- Cross-validation provides additional evaluation but does not establish clinical validity.
- Potential target leakage requires investigation into how relevant features were generated and when they become available.
- The notebook does not establish that its models generalize to real-world clinical settings.
- The project is an educational classification exercise, not a substitute for medical expertise, clinical testing, or professional diagnosis.

## Future Improvements

Potential extensions include:

- Hyperparameter tuning using cross-validation
- Additional evaluation metrics and ROC or precision-recall curves
- Feature importance and model interpretability analysis
- More detailed investigation of feature provenance and potential leakage
- External validation using an appropriate independent dataset
- Improved documentation of data sources, data quality, and experimental results

## License

No open-source license is specified for this project at present. The code and dataset may be subject to separate usage and redistribution rights. Add an appropriate license only after confirming that you have the rights to do so.

## Author

**Mohamed Sabbah**

GitHub: [YOUR_GITHUB_PROFILE](https://github.com/mohamedsabbah10)

LinkedIn: [Mohamed Sabbah](https://www.linkedin.com/in/mohamed-sabbah-60519413a/)

---

*This project is intended solely for educational and portfolio purposes. It is not a clinically validated diagnostic tool and must not be used to make medical decisions.*

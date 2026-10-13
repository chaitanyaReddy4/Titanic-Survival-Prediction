# Titanic Survival Prediction Using Logistic Regression

## Project Overview

This project predicts whether a passenger survived the Titanic disaster using **Machine Learning and Logistic Regression**. It uses passenger information such as age, sex, passenger class, fare, family relationships, and title to build a binary classification model.

The project demonstrates an end-to-end machine learning workflow, including data exploration, preprocessing, feature engineering, model training, evaluation, visualization, and prediction.

## Objectives

- Explore and understand the Titanic passenger dataset.
- Handle missing values and prepare features for machine learning.
- Perform feature engineering to improve model inputs.
- Encode categorical variables and scale numerical features.
- Train a Logistic Regression classification model.
- Evaluate performance using classification metrics and visualizations.
- Generate survival predictions and probability estimates.

## Dataset

The project uses the Titanic dataset containing **891 passenger records and 12 original columns**.

Important features include:

| Feature | Description |
|---|---|
| `Pclass` | Passenger class (1st, 2nd, or 3rd) |
| `Sex` | Passenger's sex |
| `Age` | Passenger's age |
| `SibSp` | Number of siblings or spouses aboard |
| `Parch` | Number of parents or children aboard |
| `Fare` | Passenger fare |
| `Embarked` | Port of embarkation |
| `Survived` | Target variable: 0 = No, 1 = Yes |

The dataset includes missing values, which are handled during preprocessing.

## Technologies Used

- **Python**
- **Pandas** and **NumPy** — data manipulation
- **Matplotlib** and **Seaborn** — data visualization
- **Scikit-learn** — preprocessing, model training, and evaluation
- **Jupyter Notebook** — experimentation and documentation

## Project Workflow

### 1. Data Exploration
- Inspected dataset dimensions, column names, and data types.
- Examined missing values and duplicate records.
- Analyzed survival distribution and survival rates by sex, passenger class, and age group.
- Explored relationships between passenger characteristics and survival.

### 2. Data Preprocessing
- Imputed missing numerical values using the median.
- Imputed missing categorical values using the most frequent value.
- Standardized numerical features using `StandardScaler`.
- Converted categorical features into numerical representations using `OneHotEncoder`.

### 3. Feature Engineering

Created additional features to capture passenger and family information:

- **FamilySize:** `SibSp + Parch + 1`
- **IsAlone:** Indicates whether a passenger was travelling alone.
- **Title:** Extracted from passenger names, such as Mr, Mrs, Miss, and Master.

### 4. Model Training
- Selected Logistic Regression for binary classification.
- Split the data into training and testing sets using an 80:20 ratio.
- Used stratified splitting with `random_state=42`.
- Combined preprocessing and classification in a Scikit-learn pipeline.
- Applied cross-validation and GridSearchCV for hyperparameter tuning.

### 5. Model Evaluation

Evaluated the model using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.

| Metric | Test Result |
|---|---:|
| Accuracy | 83.80% |
| ROC-AUC | 0.8758 |
| Precision — Survived class | 0.81 |
| Recall — Survived class | 0.75 |
| F1-score — Survived class | 0.78 |

**Confusion matrix:**

| | Predicted: Not Survived | Predicted: Survived |
|---|---:|---:|
| Actual: Not Survived | 98 | 12 |
| Actual: Survived | 17 | 52 |

The results show that the model correctly classified many passengers in the test set. However, some passengers were misclassified, so the model is not perfect.

### 6. Visualizations and Interpretation

The notebook includes:
- Survival distribution and exploratory data analysis plots.
- A correlation heatmap.
- A confusion matrix heatmap.
- An ROC curve.
- Logistic Regression coefficient analysis.
- A decision-boundary visualization using selected numerical features.

### 7. Sample Predictions

The trained model also generates survival predictions and estimated survival probabilities for sample passenger profiles.

These predictions are model estimates, not guarantees of actual survival outcomes.

## Results and Conclusion

The Logistic Regression model achieved **83.80% test accuracy** and a **ROC-AUC score of 0.8758** on the selected train-test split.

The project demonstrates how preprocessing, feature engineering, categorical encoding, feature scaling, and model evaluation can be combined into a reproducible machine learning pipeline.

Hyperparameter tuning was performed using GridSearchCV, but it did not improve the reported test accuracy over the baseline model.

## How to Run the Project

1. Clone or download this repository.
2. Install Python and Jupyter Notebook.
3. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

4. Ensure the dataset file is available in the notebook's working directory.
5. Open the notebook:

   ```bash
   jupyter notebook
   ```

6. Run the cells in `Titanic_Survival_Prediction.ipynb` from top to bottom.

## Future Improvements

- Compare Logistic Regression with Decision Tree, Random Forest, and other classification models.
- Explore additional feature engineering and feature selection techniques.
- Evaluate alternative classification thresholds.
- Perform further validation to assess generalization.

## Repository Structure

```text
Titanic-Survival-Prediction/
├── Titanic_Survival_Prediction.ipynb
├── titanic_dataset.csv
└── README.md
```

The dataset file should be included only if you have permission to redistribute it and intend to keep it in the repository.

## Skills Demonstrated

Python, Pandas, NumPy, Exploratory Data Analysis, Data Preprocessing, Feature Engineering, Scikit-learn, Logistic Regression, Classification Metrics, Model Evaluation, Data Visualization, and Jupyter Notebook.

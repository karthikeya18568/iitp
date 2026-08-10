# Module 2 — Analytics Pipeline

## Objective

This module follows one cohesive Titanic workflow: load the dataset once, save an offline fallback, profile and clean it, perform EDA, then continue from the same cleaned dataset into classification, imbalance comparison, hyperparameter tuning, regression, and model persistence.

## Run order

```bash
pip install -r requirements.txt
python 01_eda.py
python 02_modeling.py
```

`01_eda.py` calls `sns.load_dataset('titanic')` once when internet/cache access is available and immediately writes `titanic.csv`. If the loader is unavailable, it reads the committed CSV fallback. `02_modeling.py` reads `data/titanic_clean.csv` produced by the first stage; it does not independently call `sns.load_dataset()`.

## Missing-value rule

The measured raw missingness is:

| Column | Missing % | Strategy |
|---|---:|---|
| age | 19.87% | median imputation because it is in the 5%–30% band |
| embarked | 0.22% | drop affected rows because it is under 5% |
| deck | 77.10% | drop column because missingness is too high for reliable imputation |
| embark_town | 0.22% | retained descriptively after the corresponding embarked rows are removed |

The exact EDA calculations and interpretation are in `outputs/eda_results.md`.

## EDA requirements covered

- `info()`, `describe()`, `shape`
- missing percentages and threshold-based decisions
- age/fare histograms and box plots
- IQR outlier counts
- fare mean, median, mode and skewness interpretation
- survival by sex, pclass, and sex+pclass using boolean masks
- exactly six-column correlation matrix and heatmap
- top two absolute off-diagonal correlations
- four multivariate charts with written interpretations
- EDA-only z-score sanity check for age and fare

## Modeling requirements covered

- stratified train/test split before preprocessing
- training-only imputation/encoding/scaling through `ColumnTransformer` + `Pipeline`
- Logistic Regression, Decision Tree, Random Forest
- Decision Tree visualization with feature/class labels
- confusion matrices, accuracy, precision, recall, F1, ROC and AUC
- baseline vs `class_weight='balanced'` vs training-fold-only SMOTE
- Random Forest `GridSearchCV` over `n_estimators`, `max_depth`, `max_features`
- OOB score from `RandomForestClassifier(oob_score=True, ...)`
- multivariate linear regression predicting fare
- MAE, RMSE, R² and Adjusted R²
- residual plot and heteroscedasticity discussion
- complete fitted classification pipeline saved with `joblib.dump`
- reload test using `joblib.load` on raw feature input

See `outputs/modeling_results.md` for the generated metrics and recommendation.

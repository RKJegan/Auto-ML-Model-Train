# Model Selection

## Supported Models

### Classification
- Logistic Regression
- Decision Tree
- Random Forest
- SVM

### Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

## Selection Strategy

In automatic mode, all models are trained with `RandomizedSearchCV` and the best is selected by:
- **Classification**: highest CV accuracy
- **Regression**: lowest CV RMSE

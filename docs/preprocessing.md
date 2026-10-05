# Preprocessing

## Numeric Features
- Imputation: median
- Scaling: StandardScaler

## Categorical Features
- Cleaning: strip whitespace, blank → NaN
- Imputation: fill with 'missing'
- Encoding: OneHotEncoder (handle_unknown='ignore')

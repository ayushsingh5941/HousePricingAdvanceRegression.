import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
from tensorflow.keras import models, layers
# Importing data
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
Id = data_test['Id']
data_target = data_train['SalePrice']


# Exploring sale price
print("Skewness = ", data_train['SalePrice'].skew())
print("kurtosis = ", data_train['SalePrice'].kurt())


# Finding correlation
corr_matrix = data_train.corr(method='spearman')
fig = plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, square=True)
corr_sale = corr_matrix['SalePrice'].sort_values(ascending=False)
print(corr_sale)


# Removing target
data_train.drop(['SalePrice'], axis=1, inplace=True)


# missing data
total = data_train.isnull().sum().sort_values(ascending=False)
percent = (data_train.isnull().sum() / data_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(10))


# pre-processing and feature engineering


def feature_engineering(data):

    # Dropping feature with more tha 90% empty data
    data.dropna(thresh=len(data) * 0.8, axis=1, inplace=True)
    # Separating category and numerical features
    category_feature = data.select_dtypes(include='object').columns
    data_cat = data[category_feature].copy()
    numerical_features = data.select_dtypes(exclude='object').columns
    data_numeric = data[numerical_features].copy()

    # Filing Na of categorical values
    data_cat.fillna('No', inplace=True)
    # Filling Na of numerical values
    data_numeric['YearBuilt'].fillna(data_numeric['YearBuilt'].mode(), inplace=True)
    data_numeric['YearRemodAdd'].fillna(data_numeric['YearRemodAdd'].mode(), inplace=True)
    data_numeric.fillna(data_numeric.median(), inplace=True)

    # Adding features
    data_numeric['Age'] = (2019 - data_numeric['YearBuilt'])
    data_numeric['AgeBin'] = data_numeric['Age'] // 10
    data_numeric['HasPool'] = data_numeric['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    data_numeric['HasGarage'] = data_numeric['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    data_numeric['TotalPorch'] = data_numeric['OpenPorchSF'] + data_numeric['EnclosedPorch'] + data_numeric[
        '3SsnPorch'] + data_numeric['ScreenPorch']
    data_numeric['HasPorch'] = data_numeric['TotalPorch'].apply(lambda x: 1 if x > 0 else 0)
    data_numeric['HasFirePlace'] = data_numeric['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    data_numeric['Has2ndFloor'] = data_numeric['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    data_numeric['TotalArea'] = data_numeric['TotalBsmtSF'] + data_numeric['1stFlrSF'] + data_numeric['2ndFlrSF'] + data_numeric['GrLivArea'] + data_numeric['GarageArea']
    data_numeric['Bathrooms'] = data_numeric['FullBath'] + data_numeric['HalfBath'] * 0.5
    data_numeric['Year average'] = (data_numeric['YearRemodAdd'] + data_numeric['YearBuilt']) / 2

    # Merging Numeric and categorical data
    data = pd.concat([data_cat, data_numeric], axis=1)

    # Making conditions as category of good avg and bad
    data['OverallQualGrade'] = data['OverallQual'].apply(lambda x: 'Bad' if x < 4 else ('Average' if x >= 4 and x <= 7 else 'Best'))
    data['OverallCondGrade'] = data['OverallCond'].apply(lambda x: 'Bad' if x < 4 else ('Average' if x >= 4 and x <= 7 else 'Best'))

    # Deleting unnecesary columns
    discarded_columns = ['Id']
    data.drop(discarded_columns, axis=1, inplace=True)
    # Removing Outliers from data
    #data = data[(data['GrLivArea'] < 4600) & (data['MasVnrArea'] < 1500)]
    return data

# Important cols
cols_imp = ['OverallQual', 'GarageCars', 'YearBuilt', 'FullBath','TotalBsmtSF',
            '1stFlrSF', 'TotRmsAbvGrd', 'HasFirePlace']



# Splitting data
x_train, x_valid, y_train, y_valid = train_test_split(data_train, data_target, test_size=0.25, random_state=42)

# featured x train
x_train_featured = feature_engineering(x_train.copy())
x_valid_featured = feature_engineering(x_valid.copy())
x_test_featured = feature_engineering(data_test.copy())

# Fetching cols for categorical encoding
cols = []
for columns in x_train_featured.columns:
    if x_train_featured[columns].dtype == 'object':
        cols.append(columns)

# Encoding categorical data
cat_enc = ce.OneHotEncoder(cols=cols)
pipe = Pipeline(steps=[('cat', cat_enc)])

x_train_encoded = pipe.fit_transform(x_train_featured)
x_valid_encoded = pipe.transform(x_valid_featured)
x_test_encoded = pipe.transform(x_test_featured)
print('Encoded data shape', x_train_encoded.shape)


# Pca and Standard scaler with pipeline
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc)])
x_train_piped = pipe.fit_transform(x_train_encoded)
x_valid_piped = pipe.transform(x_valid_encoded)
x_test_piped = pipe.transform(x_test_encoded)

print('shape x_train_piped', x_train_piped.shape)

# Testing model
def test_model(reg, X_train=x_train_piped, Y_train=y_train, X_valid=x_valid_piped, Y_valid=y_valid, regname='none'):

    reg.fit(X_train, Y_train)
    pred = reg.predict(X_valid)
    print('RMLSE regressor', regname,  np.sqrt(mean_squared_log_error(Y_valid, pred)))

# models
# Xgboost
xgboost = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')

# Lgbm
"""lgb_reg = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression', num_leaves=2000,
                            learning_rate=0.007, n_estimators=3000, max_depth=6,
                            metric='rmse', bagging_fraction=0.8, feature_fraction=0.1, reg_lambda=0.9)"""
# Random forest
rf_class = RandomForestRegressor(random_state=42, n_estimators=1000, max_depth=15, max_features=0.9, bootstrap=True)

# Gradient boosting regression
gbm = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01,
                                   max_depth=3, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', alpha=0.9, random_state=42)








"""
def train_model(X_train, Y_train, X_valid, Y_valid, X_test, estimators):
    # Stacking Regressors
    stack_reg = StackingRegressor(estimators=estimators, cv=5, n_jobs=-1)
    stack_reg.fit(X_train, Y_train)
    stack_model_predict = stack_reg.predict(X_valid)
    print('RMLSE Stacked', np.sqrt(mean_squared_log_error(Y_valid, stack_model_predict)))
    stack_model_test_pred = np.expm1(stack_reg.predict(X_test))
    return stack_model_test_pred


# Training neural network for prediction
estimator = [('rnd_forest', rf_class), ('lgbm', lgb_reg), ('xgb', xgboost), ('gbm', gbm)]
pred = train_model(x_train_piped, y_train, x_valid_piped, y_valid, x_test_piped, estimator)
pred= Id
# Saving result on test set
output = pd.DataFrame({'Id': Id,
                       'SalePrice': pred})

output.to_csv(r'submission.csv', index=False)
# RMLSE Stacked 0.009505242331060799

"""

model = models.Sequential(layers.Flatten(input_shape=()))
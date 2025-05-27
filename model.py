from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

def build_pipeline(preprocessor):
    pipe = Pipeline([
        ('preprocessing', preprocessor),
        ('regressor', Ridge())  # Placeholder
    ])
    return pipe

def tune_model(pipe, X_train, y_train):
    param_grid = [
        {
            'regressor': [Ridge()],
            'regressor__alpha': [0.1, 1.0, 10.0, 100, 1000],
            'regressor__fit_intercept': [True, False],
            'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr']
        },
        {
            'regressor': [LinearRegression()],
            'regressor__fit_intercept': [True, False]
        }
    ]
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    return grid

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("R²:", r2_score(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("MAE:", mean_absolute_error(y_test, y_pred))
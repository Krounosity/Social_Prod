from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer

def build_preprocessor(num_cols, cat_cols, skewed_cols):
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False))
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ('scaler', StandardScaler(with_mean=False))
    ])

    skewed_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('power', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler(with_mean=False))
    ])

    preprocessor = ColumnTransformer([
        ('numerical', num_pipe, [col for col in num_cols if col not in skewed_cols]),
        ('categorical', cat_pipe, cat_cols),
        ('skewed', skewed_pipe, skewed_cols)
    ])

    return preprocessor
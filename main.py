import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import build_preprocessor
from model import build_pipeline, tune_model, evaluate_model
from explain import explain_model
import config

df = pd.read_csv('data/social_media_vs_productivity.csv')

df['productivity_diff'] = df['perceived_productivity_score'] - df['actual_productivity_score']
imp_cols = config.NUM_COLS + config.CAT_COLS
mask = df[config.TARGET_COL].notna()
X = df.loc[mask, imp_cols]
y = df.loc[mask, config.TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

preprocessor = build_preprocessor(config.NUM_COLS, config.CAT_COLS, ['daily_social_media_time'])
pipe = build_pipeline(preprocessor)
model = tune_model(pipe, X_train, y_train)

evaluate_model(model, X_test, y_test)
explain_model(model, X_train)

import joblib
joblib.dump(model.best_estimator_, 'best_model.pkl')
import shap
import matplotlib.pyplot as plt

def explain_model(model, X_train):
    preprocessed = model.named_steps['preprocessing'].transform(X_train)
    regressor = model.named_steps['regressor']
    explainer = shap.Explainer(regressor.predict, preprocessed)
    shap_values = explainer(preprocessed[:100])
    feature_names = model.named_steps['preprocessing'].get_feature_names_out()
    shap.summary_plot(shap_values, preprocessed[:100], feature_names=feature_names)
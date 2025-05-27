import matplotlib.pyplot as plt
import seaborn as sns

def plot_numerical_distributions(df, num_cols):
    for col in num_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True, bins=36)
        plt.title(f'{col}')
        plt.show()

def plot_categorical_distributions(df, cat_cols):
    for col in cat_cols:
        plt.figure(figsize=(12, 4))
        sns.countplot(x=col, data=df)
        plt.title(f'{col} count')
        plt.show()

def plot_correlation_heatmap(df, num_cols):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()
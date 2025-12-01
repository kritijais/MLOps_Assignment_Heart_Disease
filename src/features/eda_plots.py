import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow

def generate_and_log_eda(df: pd.DataFrame, artifact_dir: str = 'eda_artifacts'):
    """
    Generates key EDA visualizations and logs them as artifacts to MLflow.
    
    Args:
        df: The cleaned pandas DataFrame.
        artifact_dir: Local directory to temporarily save plots before logging.
    """
    # Create a temporary directory for artifacts
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    print("--- Starting EDA Visualization and MLflow Logging ---")

    # 1. Class Balance Plot (Bar Chart)
    plt.figure(figsize=(6, 5))
    sns.countplot(x='heart_disease', data=df)
    plt.title('Target Variable Class Balance (0: No Disease, 1: Disease)')
    plt.xlabel('Heart Disease Presence')
    plt.ylabel('Count')
    balance_path = os.path.join(artifact_dir, 'class_balance.png')
    plt.savefig(balance_path)
    plt.close()
    mlflow.log_artifact(balance_path, "eda_plots")
    print(f"Logged artifact: {balance_path}")

    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    # Note: Correlation matrix requires numerical data. 
    # Categorical columns must be handled, here we use only numerical parts for simplicity
    # or convert all types to numeric (which might not be ideal for true categoricals)
    
    # For a comprehensive heatmap, we temporarily convert suitable columns to numeric.
    # We will use only columns that are already numeric or binarized for the plot.
    numerical_cols = df.select_dtypes(include=['number']).columns
    corr_matrix = df[numerical_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Feature Correlation Heatmap')
    heatmap_path = os.path.join(artifact_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    mlflow.log_artifact(heatmap_path, "eda_plots")
    print(f"Logged artifact: {heatmap_path}")

    # 3. Histograms for Numerical Features
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    fig, axes = plt.subplots(len(numerical_features), 1, figsize=(8, 4 * len(numerical_features)))
    for i, col in enumerate(numerical_features):
        sns.histplot(df, x=col, hue='heart_disease', kde=True, ax=axes[i], palette='viridis')
        axes[i].set_title(f'Distribution of {col} by Disease Status')
    
    plt.tight_layout()
    hist_path = os.path.join(artifact_dir, 'numerical_histograms.png')
    plt.savefig(hist_path)
    plt.close()
    mlflow.log_artifact(hist_path, "eda_plots")
    print(f"Logged artifact: {hist_path}")
    
    print("--- EDA Visualization and MLflow Logging Complete ---")

# We will not run this script directly, it will be imported by main.py
if __name__ == '__main__':
    print("This script is intended to be imported.")
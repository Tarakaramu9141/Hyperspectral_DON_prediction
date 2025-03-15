# data_processing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path):
    """Load and preprocess hyperspectral data."""
    try:
        df = pd.read_csv('D:/Projects_for_resume/Hyperspectral_DON_prediction/hypersspectral.csv')
        logging.info("Data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        logging.warning("Missing values detected. Imputing with mean.")
        df.fillna(df.mean(), inplace=True)

    # Separate features and target
    X = df.drop(columns=['hsi_id', 'vomitoxin_ppb']).values  # Spectral bands
    y = df['vomitoxin_ppb'].values  # DON concentration in ppb

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Features normalized.")

    return X_scaled, y, scaler

def explore_data(X, y):
    """Visualize spectral data characteristics."""
    # Average reflectance over wavelengths
    plt.figure(figsize=(10, 6))
    plt.plot(np.mean(X, axis=0), label='Average Reflectance')
    plt.xlabel('Wavelength Band')
    plt.ylabel('Reflectance')
    plt.title('Average Spectral Reflectance')
    plt.legend()
    plt.savefig('avg_reflectance.png')
    plt.close()

    # Target distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(y, bins=30, kde=True)
    plt.xlabel('DON Concentration (ppb)')
    plt.title('Distribution of DON Levels')
    plt.savefig('don_distribution.png')
    plt.close()
    logging.info("Data exploration visualizations saved.")

if __name__ == "__main__":
    X, y, scaler = load_and_preprocess_data('hyperspectral_data.csv')
    explore_data(X, y)
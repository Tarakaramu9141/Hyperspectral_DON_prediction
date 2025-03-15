# evaluation.py
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from data_processing import load_and_preprocess_data
from models import build_baseline_model, build_attention_model, train_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logging.info(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    return y_pred, {'mae': mae, 'rmse': rmse, 'r2': r2}

def plot_predictions(y_test, y_pred, model_name):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, c='blue', label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual DON (ppb)')
    plt.ylabel('Predicted DON (ppb)')
    plt.title(f'{model_name} Predictions')
    plt.legend()
    plt.savefig(f'{model_name.lower()}_predictions.png')
    plt.close()

if __name__ == "__main__":
    # Load and split data
    X, y, scaler = load_and_preprocess_data('hyperspectral_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Baseline model
    baseline_model = build_baseline_model(X.shape[1])
    train_model(baseline_model, X_train, y_train, X_val, y_val)
    y_pred_baseline, baseline_metrics = evaluate_model(baseline_model, X_test, y_test)
    plot_predictions(y_test, y_pred_baseline, "Baseline")

    # Attention model
    attention_model = build_attention_model(X.shape[1])
    train_model(attention_model, X_train, y_train, X_val, y_val)
    y_pred_attention, attention_metrics = evaluate_model(attention_model, X_test, y_test)
    plot_predictions(y_test, y_pred_attention, "Attention")

    # Save weights with correct extension
    attention_model.save_weights('attention_model_weights.weights.h5')
    logging.info("Attention model weights saved as 'attention_model_weights.weights.h5'.")
# Hyperspectral DON Prediction 🌽🌾

Predict DON (Deoxynivalenol) concentrations in corn samples using hyperspectral imaging data. This project leverages machine learning to analyze spectral reflectance and provide accurate predictions.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-1.0+-green.svg?style=flat-square" alt="Streamlit Version">
  <img src="https://img.shields.io/badge/scikit_learn-1.0+-orange.svg?style=flat-square" alt="scikit-learn Version">
  <img src="https://img.shields.io/badge/TensorFlow-2.0+-red.svg?style=flat-square" alt="TensorFlow Version">
</p>

## Project Structure 📂
hyperspectral-don-prediction/
├── data_processing.py    # Handles data loading, cleaning, and preprocessing
├── models.py             # Contains the machine learning model definition and training logic
├── evaluation.py         # Evaluates the model's performance
├── app.py                # Streamlit application for interactive predictions
├── hyperspectral_data.csv # The dataset containing spectral reflectance data
├── requirements.txt      # Lists project dependencies
├── Dockerfile            # Docker configuration for containerization
└── README.md             # This file

## Getting Started 🚀

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd hyperspectral-don-prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    This will open the app in your browser, allowing you to upload spectral data and receive DON concentration predictions in real-time.

4.  **Run the evaluation script:**
    ```bash
    python evaluation.py
    ```
    This will execute the model evaluation and display performance metrics.

## Docker Deployment 🐳

1.  **Build the Docker image:**
    ```bash
    docker build -t don-prediction .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8501:8501 don-prediction
    ```
    This will make the Streamlit app accessible at `http://localhost:8501`.

## Key Features ✨

* **Interactive Streamlit App:** Upload your own hyperspectral data and get instant DON concentration predictions.
* **Modular Code:** Well-organized Python scripts for easy maintenance and future enhancements.
* **Robust Evaluation:** Comprehensive evaluation metrics and visualizations to assess model performance.
* **Containerized Deployment:** Docker support for easy deployment and portability.
* **Data Preprocessing:** Robust handling of missing data and feature normalization.
* **Machine Learning Model:** Employing neural networks for accurate predictions.

<p align="center">
  <img src="images/corn.gif" width="100" height="100">
  <img src="images/ml.gif" width="100" height="100">
</p>

## Contributing 🤝

Contributions are welcome! Feel free to submit pull requests or open issues to suggest improvements or report bugs.





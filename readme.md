# Customer Churn Predictor

## Overview
This project aims to predict customer churn using a neural network model built with TensorFlow and Keras. The application allows users to input customer data and receive a prediction on the likelihood of churn.

## Features
- User-friendly interface built with Streamlit.
- Input fields for various customer attributes.
- Real-time prediction of churn probability.
- Visualizations of input data distributions.
- Model training and evaluation using TensorFlow.

## Technologies Used
- **Python**: Programming language used for the project.
- **TensorFlow**: Framework for building and training the neural network model.
- **Keras**: High-level API for building neural networks.
- **Streamlit**: Framework for creating web applications for machine learning projects.
- **Pandas**: Data manipulation and analysis library.
- **NumPy**: Library for numerical computations.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/anubhab-m02/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For macOS/Linux
   .\venv\Scripts\activate   # For Windows
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the application in your web browser.
2. Fill in the customer details in the sidebar.
3. Click the "Predict" button to see the churn probability.
4. Review the prediction results and visualizations.

## Model Training
The model is trained using a dataset of customer information. The training process includes:
- Data preprocessing (scaling and encoding).
- Model architecture definition.
- Training with early stopping and TensorBoard for monitoring.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.




# Stock Price Prediction Using Stacked LSTM Deep Neural Network

This project aims to predict stock prices using a Stacked Long Short-Term Memory (LSTM) deep neural network model. Leveraging historical stock data obtained from the Alpha Vantage API, the project employs advanced time series analysis techniques. The dataset is meticulously partitioned into training (65%) and testing sets to facilitate robust model evaluation. Employing a combination of 100 and 150 days closing prices as independent variables, the model aims to forecast the subsequent day's stock price with a high degree of accuracy. Implementation of the model is achieved utilizing TensorFlow and Keras, while Pandas facilitates efficient data manipulation. Furthermore, the project incorporates visualization of predicted trends using Matplotlib, enhancing interpretability and aiding decision-making processes.

## Key Features

- Predicts stock prices using Stacked LSTM neural network.
- Utilizes historical stock data from Alpha Vantage API.
- Employs advanced time series analysis techniques for model training.
- Implements data preprocessing and manipulation with Pandas.
- Visualizes predicted trends using Matplotlib.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit web app:

```bash
streamlit run app.py
```

## Usage

1. Access the web app by opening the provided URL in your web browser.
2. Enter the desired stock symbol and select the prediction horizon.
3. View the predicted trends and compare them with actual prices.


Feel free to customize the content according to your project's specific details and requirements. If you have any questions or need further assistance, let me know!

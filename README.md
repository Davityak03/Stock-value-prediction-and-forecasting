# Stock-value-prediction-and-forecasting
The notebook implements a machine learning model using a Long Short-Term Memory (LSTM) neural network for stock market prediction. 

## Project Overview
This project focuses on predicting and forecasting stock market trends using historical stock data for Apple Inc. (AAPL). The project includes data retrieval, preprocessing, and the implementation of an LSTM neural network model to predict future stock prices.

## Installation
Ensure you have the necessary Python packages installed before running the notebook. The required packages are:

```bash
pip install pandas==1.5.3
pip install pandas_datareader==0.10.0
pip install tensorflow
```

## Data Retrieval
The notebook retrieves historical stock data for Apple Inc. (AAPL) from the Tiingo API using `pandas_datareader`. Ensure you have your API key ready.

```python
import pandas_datareader as pdr
from datetime import datetime

key = "API_KEY"
df = pdr.get_data_tiingo("AAPL", api_key=key)
```

## Machine Learning Model
### Data Preprocessing
The data is split into training and testing sets. The training set is used to train the LSTM model, and the testing set is used to evaluate its performance.

### Model Architecture
The model is a stacked LSTM network with three LSTM layers, each containing 50 units, followed by a dense layer with one unit to output the predicted stock price.

```python
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
```

### Model Training
The model is trained on the training data for 100 epochs with a batch size of 64.

```python
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)
```

### Model Evaluation
The model's performance is evaluated using the root mean squared error (RMSE) on both training and testing data. Predictions are also plotted against the actual stock prices for visual comparison.

```python
from sklearn.metrics import mean_squared_error
import math

train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
```

### Future Stock Price Prediction
The notebook also includes functionality to predict future stock prices for a given period.

## Visualizations
The notebook provides visualizations that compare the actual stock prices with the predicted prices to assess the model's accuracy.

## Future Work
- Implementing more advanced machine learning models for improved accuracy.
- Expanding the analysis to include multiple stocks and additional features.
- Enhancing the model to predict longer-term trends.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

#controller
epochs = 100
sequence_length = 5

# Read the CSV file
df = pd.read_csv('datasetdummy.csv')

# Select the columns of interest
df = df[['date', 'price', 'stock_name', 'PE_ratio', 'PEG_ratio', 'EPS','EPS Growth']]

# Convert 'date' to datetime
df['date'] = pd.to_datetime(df['date'])

# Normalize the features using MinMaxScaler (excluding 'stock_name' and 'date' columns)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['price', 'PE_ratio', 'PEG_ratio', 'EPS','EPS Growth']])

# The to_sequences function is used to create sequences from the scaled data
def to_sequences_eps(data, seq_length):
    x = []
    y = []
    EPS_index = data.columns.get_loc('EPS')  # Replace 'EPS' with the actual column name if different
    for i in range(len(data)-seq_length):
        x.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, EPS_index])  # Target is the EPS column
    return np.array(x), np.array(y)

sequences, labels = to_sequences(scaled_data, sequence_length)

# Split the sequences into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 5)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model using Mean Squared Error (MSE) loss function and the Adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
history = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=1)

model.save('dummy_price_prediction_model5.h5')

# Get predictions for the test set


# Assuming you have already split your data into X_train, X_test, y_train, y_test
# And your model has been trained (history = model.fit(...))

# Retrieve the last date for each test sequence
# Note that because each test sequence is `sequence_length` long,
# the date we want is `sequence_length` days after the last date of each sequence
test_dates = df['date'].values[-len(y_test) - sequence_length:-sequence_length]

# Get predictions for the test data and inverse scale them
predicted_prices = model.predict(X_test)

# Inverse scaling of the predictions and the actual prices using the scaler previously fitted
# It's necessary because the test data was scaled as well
dummy_test_prices = np.zeros(shape=(len(predicted_prices), scaled_data.shape[1]))
dummy_test_prices[:,0] = predicted_prices[:,0]  # Using the predicted prices for scaling
denormalized_predictions = scaler.inverse_transform(dummy_test_prices)[:,0]

dummy_true_prices = np.zeros(shape=(len(y_test), scaled_data.shape[1]))
dummy_true_prices[:,0] = y_test  # Using the true prices for scaling
denormalized_true_prices = scaler.inverse_transform(dummy_true_prices)[:,0]

# Print the date and its corresponding true and predicted price
for date, true, predicted in zip(test_dates, denormalized_true_prices, denormalized_predictions):
    print(f"Date: {pd.to_datetime(date).date()}, True Price: {true:.2f}, Predicted Price: {predicted:.2f}")

# Plot the actual vs predicted prices over time
plt.figure(figsize=(15,7))
plt.plot(test_dates, denormalized_true_prices, label='Actual Price', color='blue')
plt.plot(test_dates, denormalized_predictions, label='Predicted Price', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()
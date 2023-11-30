import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Controller
epochs = 100
sequence_length = 5

# Read the CSV file
df = pd.read_csv('datasetdummy.csv')

# Select the columns of interest
df = df[['date', 'price', 'stock_name', 'PE_ratio', 'PEG_ratio', 'EPS', 'EPS Growth']]

# Convert 'date' to datetime
df['date'] = pd.to_datetime(df['date'])

# Normalize the features using MinMaxScaler (excluding 'stock_name' and 'date' columns)
scaler = MinMaxScaler(feature_range=(0, 1))
features_to_scale = df[['price', 'PE_ratio', 'PEG_ratio', 'EPS', 'EPS Growth']]
scaled_data = scaler.fit_transform(features_to_scale)

# Indices after excluding 'date' and 'stock_name'
price_index, pe_ratio_index, peg_ratio_index, eps_index, eps_growth_index = range(scaled_data.shape[1])

# The to_sequences function used to create sequences targeting EPS
def to_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, eps_index])  # Targeting the EPS
    return np.array(x), np.array(y)

sequences, labels = to_sequences(scaled_data, sequence_length)

# Split the sequences into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Define a new model optimized for EPS prediction
model_eps = Sequential()
model_eps.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, scaled_data.shape[1])))
model_eps.add(Dropout(0.2))
model_eps.add(LSTM(50, return_sequences=False))
model_eps.add(Dropout(0.2))
model_eps.add(Dense(1))  # Output layer with a single neuron for regression output

# Compile the model using Mean Squared Error (MSE) loss function and the Adam optimizer
model_eps.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the EPS data
history_eps = model_eps.fit(x=X_train, y=y_train, epochs=epochs, batch_size=32,
                            validation_data=(X_test, y_test), verbose=1)

# Save the model if needed
model_eps.save('dummy_eps_prediction_model.h5')

# Get predictions for the test set
predicted_eps = model_eps.predict(X_test)

# Prepare a dummy array with zeros to inverse the scaling
dummy_test_eps = np.zeros(shape=(len(predicted_eps), scaled_data.shape[1]))
dummy_test_eps[:, eps_index] = predicted_eps[:, 0]  # Put the predicted EPS

# Inverse transformation to get the actual scale of EPS
denormalized_predicted_eps = scaler.inverse_transform(dummy_test_eps)[:, eps_index]

# Perform the same inverse scaling for the true EPS values
dummy_true_eps = np.zeros(shape=(len(y_test), scaled_data.shape[1]))
dummy_true_eps[:, eps_index] = y_test

# Inverse transformation to get the actual scale of true EPS values
denormalized_true_eps = scaler.inverse_transform(dummy_true_eps)[:, eps_index]

# Print the date and its corresponding true and predicted EPS
test_dates = df['date'].values[-len(y_test) - sequence_length:-sequence_length]
for date, true_eps, predicted_eps in zip(test_dates, denormalized_true_eps, denormalized_predicted_eps):
    print(f"Date: {pd.to_datetime(date).date()}, True EPS: {true_eps:.2f}, Predicted EPS: {predicted_eps:.2f}")

plt.figure(figsize=(15,7))
plt.plot(test_dates, denormalized_true_eps, label='Actual EPS', color='blue')
plt.plot(test_dates, denormalized_predicted_eps, label='Predicted EPS', color='red')
plt.xlabel('Date')
plt.ylabel('EPS')
plt.title('Stock EPS Prediction')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()
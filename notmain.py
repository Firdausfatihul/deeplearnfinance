import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model (assuming it is saved as 'stock_price_prediction_model.h5')
model = load_model('dummy_price_prediction_model20.h5')

# Load dataset B into a DataFrame
df_B = pd.read_csv('datasetdummy2.csv')
df_B['date'] = pd.to_datetime(df_B['date'])

# Normalize dataset B using the MinMaxScaler from dataset A
# Note: It's crucial to use the same scaler object that was fitted on dataset A to ensure consistency
# If the scaler was not saved from the last session, refit it on the training dataset of A and save it
scaler_B = MinMaxScaler(feature_range=(0, 1))
# Assuming that df_A is your original training set and you are re-fitting the scaler
df_A = pd.read_csv('datasetdummy.csv')
scaler_B.fit(df_A[['price', 'PE_ratio', 'PEG_ratio', 'EPS','EPS Growth']])
scaled_data_B = scaler_B.transform(df_B[['price', 'PE_ratio', 'PEG_ratio', 'EPS','EPS Growth']])

def to_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])  # Price is the first column and our target
    return np.array(x), np.array(y)

# Create sequences for dataset B
sequence_length = 20  # Same sequence length that was used for dataset A
sequences_B, _ = to_sequences(scaled_data_B, sequence_length)

# Make predictions using the trained model on the new dataset B sequences
predicted_B = model.predict(sequences_B)

def predict_sequence(model, initial_sequences, length, original_data):
    predictions = []

    # Start with the initial sequence
    current_sequence = initial_sequences[0].copy()

    for i in range(length):
        # Reshape the sequence to match the model's input format
        sequence_reshaped = np.array([current_sequence])
        
        # Predict the next step (price)
        next_step_prediction = model.predict(sequence_reshaped)

        # Append the prediction to the output list
        predictions.append(next_step_prediction[0, 0])
        
        # Prepare the next sequence
        if i < length - 1:
            # Take the next sequence from the original data
            next_sequence = initial_sequences[i + 1].copy()
            # Replace its price with the predicted price
            next_sequence[0, 0] = next_step_prediction[0, 0]
            current_sequence = next_sequence

    return np.array(predictions)

# Predict a sequence of desired length
predicted_length = len(sequences_B)  # or any other length you want
predicted_sequence = predict_sequence(model, sequences_B, predicted_length, scaled_data_B)

# Invert normalization for the predicted sequence
dummy_array_B = np.zeros(shape=(len(predicted_sequence), scaled_data_B.shape[1]))
dummy_array_B[:, 0] = predicted_sequence
denormalized_predictions_B = scaler_B.inverse_transform(dummy_array_B)[:, 0]

# The rest of the plotting code remains the same

true_prices_B = df_B['price'].values[sequence_length:]
##true_pe_B = df_B['PE_ratio'].values[sequence_length:]
##true_peg_B = df_B['PEG_ratio'].values[sequence_length:]



# Retrieve corresponding dates for sequences in dataset B
# The date for each prediction should be the date following the end of the sequence it was predicted on
# Hence, we get dates starting from the 'sequence_length' + 1 to the end of the DataFrame
dates_B = df_B['date'].values[sequence_length:]

# Plot predicted and true prices for dataset B
plt.figure(figsize=(15,7))
plt.plot(dates_B, denormalized_predictions_B, label='Predicted Price')
plt.plot(dates_B, true_prices_B, label='True Price', alpha=0.7)  # Alpha for transparency
##plt.plot(dates_B, true_pe_B, label='True PE', alpha=0.7)  # Alpha for transparency
##plt.plot(dates_B, true_peg_B, label='True PEG', alpha=0.7)  # Alpha for transparency
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Predicted vs True Stock Prices for Dataset B')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
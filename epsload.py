import pandas as pd
import numpy as np
import joblib
from keras.models import load_model

# Load the pre-trained EPS prediction model
model_eps = load_model('dummy_eps_prediction_model.h5')

# Load the scaler
scaler = joblib.load('scaler.gz')

# Read the CSV file
df = pd.read_csv('datasetdummy.csv')
df['date'] = pd.to_datetime(df['date'])

# Normalize the features using the loaded scaler (excluding 'stock_name' and 'date' columns)
features_to_scale = df[['price', 'PE_ratio', 'PEG_ratio', 'EPS', 'EPS Growth']]
scaled_data = scaler.transform(features_to_scale)

# Use the to_sequences function (as defined in your training script) to prepare the full timeframe data
sequences, _ = to_sequences(scaled_data, sequence_length)

# Perform prediction for the full timeframe
predicted_eps = model_eps.predict(sequences)

# Prepare dummy arrays for inverse scaling the predicted EPS values
dummy_predicted_eps = np.zeros(shape=(len(predicted_eps), scaled_data.shape[1]))
dummy_predicted_eps[:, eps_index] = predicted_eps[:, 0]

# Inverse scaling of the predictions to get EPS in its original scale
denormalized_predicted_eps = scaler.inverse_transform(dummy_predicted_eps)[:, eps_index]

# Output the predictions
output_df = pd.DataFrame({
    'Date': df['date'].values[sequence_length:],
    'Predicted_EPS': denormalized_predicted_eps
})

# Optionally, save the predictions to a CSV file
output_df.to_csv('predicted_eps.csv', index=False)

print(output_df.head())  # Print the first few predictions
# For the purpose of this example, let's assume that we have a dataset preloaded.
# In practice, you would load your dataset from a file or a database.

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import Input, Dense, Dropout, Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam   
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Assume our dataset is called `df` and has the columns:
# 'volume', 'pe_ratio', 'dividend_yield', 'std_dev_price', 'cum_date_price'
# df = pd.read_csv('path_to_dataset.csv')

# Placeholder dataset generation
np.random.seed(0)
num_samples = 1000
df = pd.DataFrame({
    'volume': np.random.rand(num_samples),
    'pe_ratio': np.random.rand(num_samples) * 20,
    'dividend_yield': np.random.rand(num_samples) * 5,
    'std_dev_price': np.random.rand(num_samples),
    'cum_date_price': np.random.rand(num_samples) * 100 + 50
})

# Splitting the dataset into features (X) and target (y)
X = df.drop('cum_date_price', axis=1).values
y = df['cum_date_price'].values.reshape(-1, 1)

# Normalizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

# Define the Mixture Density Network (MDN)
num_components = 10  # This is a hyperparameter representing the number of distributions in the mixture

def build_mdn(input_shape, num_components):
    inputs = Input(shape=(input_shape,))
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    pi = Dense(num_components, activation='softmax', name='pi')(x)
    mu = Dense(num_components, name='mu')(x)
    sigma = Dense(num_components, activation='exponential', name='sigma')(x)
    mdn_outputs = tf.keras.layers.concatenate([pi, mu, sigma], axis=-1)
    model_outputs = split_mdn_outputs(num_components)(mdn_outputs)
    model = Model(inputs, mdn_outputs)
    return model

def split_mdn_outputs(num_components):
    def split(x):
        pi, mu, sigma = tf.split(x, num_or_size_splits=[num_components, num_components, num_components], axis=-1)
        return [pi, mu, sigma]
    return Lambda(split)

# Loss function for MDN
def mdn_loss(y_true, y_pred):
    pi, mu, sigma = tf.split(y_pred, num_or_size_splits=3, axis=1)
    mixture_distribution = tfp.distributions.Categorical(probs=pi)
    components_distribution = tfp.distributions.Normal(loc=mu, scale=sigma)
    mdn_distribution = tfp.distributions.MixtureSameFamily(
        mixture_distribution=mixture_distribution,
        components_distribution=components_distribution)
    
    log_likelihood = mdn_distribution.log_prob(y_true)
    return -tf.reduce_mean(log_likelihood)

# Function to calculate the probability of a specific target price
def compute_price_probability(target_price, pi, mu, sigma):
    mixture_distribution = tfp.distributions.Categorical(probs=pi)
    components_distribution = tfp.distributions.Normal(loc=mu, scale=sigma)
    mdn_distribution = tfp.distributions.MixtureSameFamily(
        mixture_distribution=mixture_distribution,
        components_distribution=components_distribution)
    
    # Calculate the cumulative distribution function value for the target price
    cdf_value = mdn_distribution.cdf(target_price)
    
    return cdf_value.numpy()  # Return the probability as a numpy array value

def compute_cumulative_probability(target_prices, pi, mu, sigma):
    mixture_distribution = tfp.distributions.Categorical(probs=pi)
    components_distribution = tfp.distributions.Normal(loc=mu, scale=sigma)
    mdn_distribution = tfp.distributions.MixtureSameFamily(
        mixture_distribution=mixture_distribution,
        components_distribution=components_distribution)
    
    # Calculate the cumulative distribution function values for all target prices
    cdf_values = mdn_distribution.cdf(target_prices[:, np.newaxis])
    
    return cdf_values.numpy()

# Hyperparameters for training
learning_rate = 0.001
batch_size = 32
max_epochs = 200

# Check if a saved model exists, if not, build and train a new one
model_path = '/mnt/data/mdn_stock_price_model.h5'

if os.path.exists(model_path):
    # Load the pre-trained model
    print("Loading saved model...")
    model = tf.keras.models.load_model(model_path, custom_objects={'mdn_loss': mdn_loss})

    # Verify the input shape
    print("Shape of X_test:", X_test.shape)

    # Get the predictions from the model
    predictions = model.predict(X_test)

    # Verify the output shape
    print("Predictions shape:", predictions.shape)

    # Ensure that the output is indeed three separate arrays

    pi_pred = predictions[:, :num_components]
    mu_pred = predictions[:, num_components:2*num_components]
    sigma_pred = predictions[:, 2*num_components:]

    # Check shapes after unpacking
    print("Shapes after unpacking: pi_pred:", pi_pred.shape, ", mu_pred:", mu_pred.shape, ", sigma_pred:", sigma_pred.shape)

    # Now verify each of the unpacked predictions
    print("pi_pred shape:", pi_pred.shape)
    print("mu_pred shape:", mu_pred.shape)
    print("sigma_pred shape:", sigma_pred.shape)

    # Choose a range of prices you're interested in
    sample_index = 0

    min_price = 0
    max_price = 200
    num_prices = 1000
    num_points = 1000
    target_prices = np.linspace(min_price, max_price, num_points)
    cumulative_probabilities = compute_cumulative_probability(
    target_prices,
    pi_pred[sample_index],
    mu_pred[sample_index],
    sigma_pred[sample_index]
)

    # Calculate the probability for each target price

    # Visualize these probabilities
    for price, prob in zip(target_prices, cumulative_probabilities):
        plt.plot(price, prob, label=f'Price: {price:.2f}')

    plt.xlabel('Target Price ($)')
    plt.ylabel('Probability')
    plt.title('Probability Distribution for Target Prices')
    plt.legend()
    plt.show()
else:
    # Build a new model
    print("Building a new model...")
    model = build_mdn(X_train.shape[1], num_components)
    model.compile(optimizer=Adam(learning_rate), loss=mdn_loss)
    
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True)
    
    # Start training
    print("Starting training...")
    history = model.fit(
        X_train, 
        y_train, 
        epochs=max_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Predict mixture components from the MDN for the test set
    pi, mu, sigma = model.predict(X_test)

    # Choose a range of prices you're interested in
    min_price = 0
    max_price = 200
    num_prices = 1000
    target_prices = np.linspace(min_price, max_price, num_prices)  # Define your range and number of prices

    # Calculate the probability for each target price
    probabilities = [compute_price_probability(price, pi, mu, sigma) for price in target_prices]

    # Visualize these probabilities
    for price, prob in zip(target_prices, probabilities):
        plt.plot(price, prob, label=f'Price: {price:.2f}')

    plt.xlabel('Target Price ($)')
    plt.ylabel('Probability')
    plt.title('Probability Distribution for Target Prices')
    plt.legend()
    plt.show()    

# Assuming `model` is your trained MDN model and `X_test` is your input features dataset







# At this point, the model is either loaded or trained and saved. We can use it for predictions or analysis.

# Please note that actual paths, model architecture, dataset and feature engineering steps should be customized.
# This code is a template to illustrate the structure of such a process.
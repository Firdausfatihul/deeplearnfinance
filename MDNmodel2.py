import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from tensorflow_probability import distributions as tfd
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import simps

def create_mockup_dataset():
    dates = pd.date_range(start='2015-03-01', end='2019-04-30', freq='B')  # Business days
    prices = np.linspace(100, 200, len(dates))  # Simulating a stock price increase
    volume = np.random.uniform(1000, 5000, len(dates))
    pe_ratio = np.random.uniform(10, 20, len(dates))
    dividend_yield = np.random.uniform(0.02, 0.05, len(dates))
    std_dev = np.random.uniform(1, 5, len(dates))

    # Convert string dates to Timestamps
    cum_dates = pd.to_datetime(['2015-03-04', '2015-03-06', '2018-03-01', '2019-04-01'])

    # Adding some noise around specific dates (cum dates)
    for date in cum_dates:
        if date in dates:
            idx = dates.get_loc(date)
            yield_value = dividend_yield[idx]
            if yield_value > 0.036:
                # The higher the yield, the greater the increase
                increase_factor = 1 + (yield_value - 0.036) * 100 # Adjust this formula as needed
                prices[idx:idx+5] *= np.linspace(increase_factor, 1, 5)  # Increasing price
            else:
                prices[idx:idx+5] *= np.linspace(1.02, 0.98, 5)  # Minor fluctuation

    data = pd.DataFrame({
        'Date': dates,
        'Volume': volume,
        'PE_Ratio': pe_ratio,
        'Dividend_Yield': dividend_yield,
        'Std_Dev_Price': std_dev,
        'Stock_Price': prices
    })

    return data

# Create and inspect the mockup dataset
dataset = create_mockup_dataset()
print(dataset)

# Feature columns
feature_cols = ['Volume', 'PE_Ratio', 'Dividend_Yield', 'Std_Dev_Price']
target_col = 'Stock_Price'

# Normalization
scaler = StandardScaler()
features = scaler.fit_transform(dataset[feature_cols])
target = dataset[target_col].values

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

def build_mdn(input_shape, num_components):
    inputs = Input(shape=input_shape)
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    pi = Dense(num_components, activation='softmax', name='pi')(x)
    mu = Dense(num_components, name='mu')(x)
    sigma = Dense(num_components, activation='exponential', name='sigma')(x)

    outputs = tf.keras.layers.concatenate([pi, mu, sigma], axis=1)

    model = Model(inputs, outputs)
    return model

# Loss Function
def mdn_loss(y_true, y_pred):
    num_components = 3  # or however many components you have

    # Extract pi, mu, sigma from y_pred
    pi = y_pred[:, :num_components]
    mu = y_pred[:, num_components:2*num_components]
    sigma = y_pred[:, 2*num_components:]

    mixture_distribution = tfd.Categorical(probs=pi)
    components_distribution = tfd.Normal(loc=mu, scale=sigma)
    mdn_distribution = tfd.MixtureSameFamily(
        mixture_distribution=mixture_distribution, 
        components_distribution=components_distribution)
    
    log_likelihood = mdn_distribution.log_prob(tf.transpose(y_true))
    return -tf.reduce_mean(log_likelihood)

num_components = 3  # Example number of components in the mixture
input_shape = x_train.shape[1:]

model = build_mdn(input_shape, num_components)
model.compile(optimizer=Adam(learning_rate=0.001), loss=mdn_loss)
history = model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.15)

def predict_and_visualize(model, x_input, num_components, num_samples=1000):
    # Predict the combined output
    y_pred = model.predict(x_input)

    # Split the predictions into pi, mu, and sigma
    pi = y_pred[:, :num_components]
    mu = y_pred[:, num_components:2*num_components]
    sigma = y_pred[:, 2*num_components:]

    samples = []

    for i in range(len(mu)):
        # For each input, draw samples from the mixture model
        sample = np.array([])
        for j in range(num_samples):
            # Choose a component based on pi (mixing coefficients)
            component = np.random.choice(num_components, p=pi[i])
            # Sample from the chosen component
            sample = np.append(sample, np.random.normal(mu[i][component], sigma[i][component]))
        samples.append(sample)

    # Visualization
    fig, axes = plt.subplots(len(x_input), 1, figsize=(8, 4 * len(x_input)))
    if len(x_input) == 1:
        axes = [axes]  # Ensure axes is iterable for a single plot

    for i, ax in enumerate(axes):
        ax.hist(samples[i], bins=50, density=True, alpha=0.6, color='blue')
        ax.set_title(f'Predicted Distribution for Input {i+1}')
        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('Density')

    plt.tight_layout()
    plt.show()


    return samples

# Example usage
samples = predict_and_visualize(model, x_test[:5], num_components)

def plot_predictions(samples, actual_values):
    for i, sample in enumerate(samples):
        plt.hist(sample, bins=50, density=True, alpha=0.5, label=f'Sample {i+1}')
        plt.axvline(x=actual_values[i], color='r', linestyle='--', label='Actual Value')
        plt.legend()
        plt.show()

plot_predictions(samples, y_test[:5])

def get_probable_price_range(mu, sigma, pi, num_std_dev=2):
    # Select the component with the highest mixing coefficient
    component = np.argmax(pi[0])
    mean = mu[0][component]
    std_dev = sigma[0][component]

    # Calculate the range around the mean within a certain number of standard deviations
    range_min = max(mean - std_dev * num_std_dev, 0)  # Ensure the minimum is not negative
    range_max = mean + std_dev * num_std_dev

    return np.linspace(range_min, range_max, 100)  # 100 points within the range

def calculate_probabilities(model, x_input, num_components):
    y_pred = model.predict(x_input)
    pi, mu, sigma = y_pred[:, :num_components], y_pred[:, num_components:2*num_components], y_pred[:, 2*num_components:]

    price_range = get_probable_price_range(mu, sigma, pi)

    probabilities = []
    for price in price_range:
        pdf_values = [pi[0][i] * norm.pdf(price, mu[0][i], sigma[0][i]) for i in range(num_components)]
        total_density = np.sum(pdf_values)
        probabilities.append(total_density)

    # Integrate the probability densities to get probabilities
    total_probability = simps(probabilities, price_range)
    probabilities_percentage = [prob / total_probability * 100 for prob in probabilities]

    return price_range, probabilities_percentage


price_range, probabilities_percentage = calculate_probabilities(model, x_test[0:1], num_components)

# Plotting the probabilities
plt.plot(price_range, probabilities_percentage)
plt.xlabel('Price')
plt.ylabel('Probability (%)')
plt.title('Probability Distribution of Stock Price')
plt.show()

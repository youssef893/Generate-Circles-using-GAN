import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.models import Input, Model
from sklearn.metrics import mean_squared_error

HIDDEN_SIZE = 30


def read_dataset(filename):
    dataset = pd.read_csv(filename)
    return dataset


def split_data(dataset):
    train = dataset[0:17]
    test = dataset[17:]
    validation = train[13:17]
    train = train[0:13]
    return train.to_numpy(), validation.to_numpy(), test.to_numpy()


def build_auto_encoder(train, validation):
    size = len(train[0])
    inputs = Input(shape=size)
    encoder = Dense(HIDDEN_SIZE, activation=LeakyReLU(alpha=0.6))(inputs)
    decoder = Dense(size, activation=LeakyReLU(alpha=0.2))(encoder)
    auto_encoder = Model(inputs=inputs, outputs=decoder)
    auto_encoder.compile(loss='mean_squared_error', optimizer='adam')
    auto_encoder.fit(train, train, validation_data=(validation, validation), epochs=1000, batch_size=256)
    encoder = Model(inputs, encoder)
    return auto_encoder, encoder


def build_decoder(size, auto_encoder):
    inputs = Input(shape=size)
    decoder = auto_encoder.layers[-1]
    decoding_model = Model(inputs, decoder(inputs))
    return decoding_model


def plot_results(predictions, test):
    n = 5  # How many digits we will display
    predictions = pd.DataFrame(predictions)
    test = pd.DataFrame(test)
    plt.figure(figsize=(20, 10))
    for i in range(n):
        # Display original
        original_X_points, original_y_points = test.iloc[i][::2], test.iloc[i][1::2]
        print(original_X_points.shape, original_y_points.shape)
        ax = plt.subplot(2, n, i + 1)
        plt.xlim([-150, 150])
        plt.ylim([-150, 150])
        plt.title("Zero")
        plt.scatter(original_X_points, original_y_points, color='indigo')

        # Display reconstruction

        predicted_X_points, predicted_y_points = predictions.iloc[i][::2], predictions.iloc[i][1::2]
        ax = plt.subplot(2, n, i + 1 + n)
        plt.xlim([-150, 150])
        plt.ylim([-150, 150])
        plt.title("Zero")
        plt.scatter(predicted_X_points, predicted_y_points, color='navy')

    plt.show()


def main():
    dataset = read_dataset('Geom.csv')
    dataset = dataset / 145
    train, validation, test = split_data(dataset)
    print(train.shape)
    print(test.shape)
    auto_encoder, encoder = build_auto_encoder(train, validation)
    predictions = auto_encoder.predict(test)
    print(mean_squared_error(test, predictions))
    decoder = build_decoder(HIDDEN_SIZE, auto_encoder)
    codes = encoder.predict(test)
    predictions = decoder.predict(codes)
    print(predictions.shape)
    print(mean_squared_error(test, predictions))
    predictions = predictions * 145
    test = test * 145
    plot_results(predictions, test)


if __name__ == '__main__':
    main()

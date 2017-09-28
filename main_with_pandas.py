import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def construct_neural_network():
    nn = Sequential()
    nn.add(Dense(8, input_dim=22, activation='relu'))
    nn.add(Dense(12, activation='relu'))
    nn.add(Dense(1, activation='sigmoid'))

    nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return nn


def read_data_from_csv(train_size: int) -> tuple:
    train_data = pd.read_csv('train.csv', index_col='Id')

    features_names = [
        'Duration',
        # 'Language',
        # 'Country',
        # 'Rating',
        'Action',
        'Adventure',
        'Animation',
        'Biography',
        'Crime',
        'Documentary',
        'Drama',
        'Family',
        'Fantasy',
        'History',
        'Horror',
        'Music',
        'Musical',
        'Mystery',
        'News',
        'Romance',
        'Sci-Fi',
        'Sport',
        'Thriller',
        'War',
        'Western',
        # 'Poster'
    ]
    print(len(features_names))
    train_features = train_data[features_names][:train_size].values
    train_labels = train_data['Target'][:train_size].values

    test_features = train_data[features_names][train_size:].values
    test_labels = train_data['Target'][train_size:].values

    return (train_features,
            train_labels,
            test_features,
            test_labels)


def start():
    np.random.seed(33)

    train_features, train_labels, test_features, test_labels = read_data_from_csv(train_size=3500)

    model = construct_neural_network()
    model.fit(train_features, train_labels, epochs=100, batch_size=512)

    predicted_test_labels = model.predict_classes(test_features)

    matches = [predicted_test_labels[i] == test_labels[i] for i in range(len(test_labels))].count(True)
    print('\n\n{} out of {}. Accuracy = {:.2f}%'.format(matches, len(test_labels), matches/len(test_labels)*100))


if __name__ == '__main__':
    start()




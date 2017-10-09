import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
import numpy as np
import csv


def construct_neural_network(input_size: int):
    nn = Sequential()
    nn.add(Dropout(0.2, input_shape=(input_size,)))
    nn.add(Dense(12, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    nn.add(Dropout(0.2))
    nn.add(Dense(18, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    nn.add(Dense(1, activation='sigmoid'))

    nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return nn


def normalize_duration(df: 'pd.DataFrame') -> 'pd.DataFrame':
    result = df.copy()
    max_value = df['Duration'].max()
    min_value = df['Duration'].min()
    result['Duration'] = (df['Duration'] - min_value) / (max_value - min_value)
    return result


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
        # 'Poster',
    ]


def add_numerical_cols_from_categorical(df: 'pd.DataFrame', col_name: str) -> 'pd.DataFrame':
    languages = pd.get_dummies(df[col_name])
    result = pd.concat([df, languages], axis=1)
    global features_names
    features_names += list(languages)
    return result


def read_data_from_csv(train_size: int) -> tuple:
    train_data = pd.read_csv('train.csv', index_col='Id')

    #shuffle rows
    train_data = train_data.iloc[np.random.permutation(len(train_data))]

    test_data = pd.read_csv('test.csv', index_col='Id')

    # merge train and test into one table,
    # one hot encoded features not present in one of the tables will be 0 in this table
    merged_data = pd.concat([train_data, test_data], axis=0)

    merged_data = normalize_duration(merged_data)

    for feature in ['Language', 'Country', 'Rating']:
        merged_data = add_numerical_cols_from_categorical(merged_data, feature)
    print(merged_data)

    merged_data = merged_data.drop(['Language', 'Country', 'Rating', 'Poster'], axis=1)

    train_features = merged_data[:train_size]
    del train_features['Target']
    train_features = train_features.values
    # print(train_features)
    train_labels = merged_data['Target'][:train_size].values

    # used for cross-validation
    # test_features = train_data[features_names][train_size:].values
    # test_labels = train_data['Target'][train_size:].values

    test_features = merged_data[3636:]
    del test_features['Target']
    test_features = test_features.values
    test_labels = None

    return (train_features,
            train_labels,
            test_features,
            test_labels)


def start():
    np.random.seed(33)

    train_features, train_labels, test_features, test_labels = read_data_from_csv(train_size=3500)

    model = construct_neural_network(train_features.shape[1])
    model.fit(train_features, train_labels, epochs=100, batch_size=512)

    # predicted_test_labels = model.predict_classes(test_features)
    #
    # matches = [predicted_test_labels[i] == test_labels[i] for i in range(len(test_labels))].count(True)
    # print('\n\n{} out of {}. Accuracy = {:.2f}%'.format(matches, len(test_labels), matches/len(test_labels)*100))

    result_prediction = model.predict(test_features)
    with open('Tsukanov_Alexander_task_2_prediction.csv', 'x') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['', 'Id', 'Probability'])
        for i in range(0, 908):
            csvwriter.writerow([i, 3636+i, float(result_prediction[i])])


if __name__ == '__main__':
    start()




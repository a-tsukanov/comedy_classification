from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np
import csv


def get_dataset_from_csv(csv_path: str,
                         x_indices: list,
                         y_index: int) -> tuple:
    """Proceeds data from csv file and writes it to a tuple of numpy arrays

    :param csv_path:
    :param x_indices: a list of features' indices
    :param y_index: index of label
    :return: a tuple (x, y) where:
    x - two-dimensional numpy array of features
    y - one-dimensional numpy array of labels
    """
    with open(csv_path, newline='') as csvfile:
        datareader = csv.reader(csvfile)
        dataset_x = []
        dataset_y = []
        for line in list(datareader):
            dataset_x.append([line[i] for i in x_indices])
            dataset_y.append(line[y_index])
        del dataset_x[0], dataset_y[0]  # deletes the first row of data which is a header
    x = np.asarray(dataset_x)
    # TODO: normalize duration
    y = np.asarray(dataset_y)
    return x, y


def construct_model():
    model = Sequential()
    model.add(Dense(8, input_dim=22, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    training_data_x, training_data_y = get_dataset_from_csv('train.csv', x_indices=[2, *list(range(6, 27))], y_index=1)

    training_data_x, test_data_x = training_data_x[:3500], training_data_x[3500:]
    training_data_y, test_data_y = training_data_y[:3500], training_data_y[3500:]
    # TODO: add string data as new boolean features
    model = construct_model()
    model.fit(training_data_x, training_data_y, epochs=100, batch_size=512)

    results = np.array(model.predict_classes(test_data_x))
    matches = 0
    for i in range(135):
        if results[i] == test_data_y[i]:
            matches += 1

    print(matches)

    # scores = model.evaluate(test_data_x, test_data_y)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

if __name__ == '__main__':
    main()


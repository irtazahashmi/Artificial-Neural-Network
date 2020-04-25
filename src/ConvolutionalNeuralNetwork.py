import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def plot_learning_curve(data, epoch):
    # Plot training and validation accuracy values
    # If validation accuracy always above training accuracy -> Not over fitting the model
    epoch_range = range(1, epoch + 1)
    plt.plot(epoch_range, data.history['accuracy'])
    plt.plot(epoch_range, data.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training and validation loss values
    # If validation loss less than training loss -> Good
    # Else keep training model
    epoch_range = range(1, epoch + 1)
    plt.plot(epoch_range, data.history['loss'])
    plt.plot(epoch_range, data.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


if __name__ == "__main__":

    # Load data
    cancer_data = datasets.load_breast_cancer()
    X = pd.DataFrame(data=cancer_data.data, columns=cancer_data.feature_names)

    # 0 is malignant and 1 is benign
    y = cancer_data.target

    # 20% test data. Stratify will split the data symmetrically in train and test set.
    training_feature, test_feature, training_target, test_target = \
        train_test_split(X, y, test_size=.2, random_state=0, stratify=y)

    # Convert pandaDF into numpy array
    scaler = StandardScaler()
    training_feature = scaler.fit_transform(training_feature)
    test_feature = scaler.transform(test_feature)

    # Now we have a 2D data but CNN accepts 3D data.
    training_feature = training_feature.reshape(455, 30, 1)
    test_feature = test_feature.reshape(114, 30, 1)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(30, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    data = model.fit(training_feature, training_target, epochs=50, validation_data=(test_feature, test_target), verbose=1)

    plot_learning_curve(data, 50)





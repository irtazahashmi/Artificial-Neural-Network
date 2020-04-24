import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# Plot a certain image
def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()


# Plot first 25 images to see data is in correct format
def verify_data_format():
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(training_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[training_labels[i]])
    plt.show()


if __name__ == "__main__":

    # Load data
    (training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    # Labels
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Turn into np array
    training_images = np.asarray(training_images)
    training_labels = np.asarray(training_labels)
    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)
    class_names = np.asarray(class_names)

    # Building the model.
    # The first layer is flatten: transforms the images from 2d to 1d. Taking all the pixels and putting them
    # in a line. This layer just reformats the data, doesn't learn anything.
    # The next two layers are fully (densely) connected neural layers. The first layer has 128 nodes and the
    # second one has 10. Each node contains a score that indicates the current image belongs to one of the 10 classes.
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Pass in the training images and labels to train the data with 10 epochs.
    model.fit(training_images, training_labels, epochs=10)

    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_accuracy)



    # ANALYSIS

    # Adding an additional layer that converts logits into probabilities.
    prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = prob_model.predict(test_images)

    # Image One -> Returns class 9 "Ankle boot"
    predictions_of_image_one = predictions[0]
    most_likely_class = np.argmax(predictions[0])









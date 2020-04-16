from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from termcolor import cprint
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Use fashion_mnist dataset from keras.
fashion_mnist = keras.datasets.fashion_mnist

# Load fashion_mnist data to create train and test datasets.
(train_images, train_classes), (test_images, test_classes) = fashion_mnist.load_data()

# Classes of clothes.
class_names = ['Футболка/топ', 'Штаны', 'Ковта', 'Платье', 'Куртка',
               'Туфля', 'Рубашка', 'Кросовок', 'Сумка', 'Ботинок']

# Divide every dataset by 255 ( Amount of pixels in every image ).
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create a model with 128 neurons in second stack and 10 neurons in third stack.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model with accuracy metrics.
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model by 10 epochs.
model.fit(train_images, train_classes, epochs=10)

# Get the accuracy of the model.
test_loss, test_acc = model.evaluate(test_images, test_classes, verbose=2)
print("\nТочность на проверочных данных: ", test_acc)

# Predict the clothes class.
predictions = model.predict(test_images)

# Function to display the clothes image and the prediction.
def plot_image(i, predictions_array, true_label, img):
    # Choose the image, prediction class and correct class with given ID ( i ).
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]

    # Basic PyPlot settings ;)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary) # Display the clothes image.

    # Set color to blue if prediction is correct, otherwize set color to red.
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    # Display the line with all important info.
    plt.xlabel("{} {:2.0f}% ({})".format(
                                        class_names[predicted_label], # Display the predicted class.
                                        100*np.max(predictions_array), # Display the confidence percentage.
                                        class_names[true_label] # Display the correct class of the clothes.
                                        ),
                                        color = color # Set color to blue ( If correct ) or red ( If not ).
    )

# Function to display 3 bars with prediction class, correct class and other guesses.
def plot_value_array(i, predictions_array, true_label):
    # Choose the image, prediction and correct class with given ID ( i ).
    predictions_array, true_label = predictions_array[i], true_label[i]

    # Basic PyPlot settings ;)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    thisplot = plt.bar(range(10), predictions_array, color="#777777") # Set other guesses bar color to gray.
    plt.ylim([0, 1]) # Set the 'y' axis limits to 0 and 1.
    predicted_label = np.argmax(predictions_array) # Get the most confident prediction.

    thisplot[predicted_label].set_color('red') # Set color of predicted class to red.
    thisplot[true_label].set_color('blue') # Set color of correct class to blue.

# Display first 30 images from train dataset.
plt.figure(figsize=(10, 10))
for i in range(30):
    plt.subplot(6, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_classes[i]])
plt.show()

# Let user choose the image.
cprint("\nВведите номер изображения:", "blue")
i = int(input(">>> ")) # Get image ID.

# Display the prediction for only one image.
img = test_images[1]
img = (np.expand_dims(img,0))

predictions_single = model.predict(img)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_classes, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions,  test_classes)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
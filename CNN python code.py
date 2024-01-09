

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import os

train_dir=r'C:\Users\USER\Desktop\train'
test_dir=r'C:\Users\USER\Desktop\testset'
validation_dir=r'C:\Users\USER\Desktop\val'

# generating batches of tensor image data
train_datagen=ImageDataGenerator(
    rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory(train_dir,
                                                 target_size=(128,128),
                                                 batch_size=64,
                                                 class_mode='categorical')
validation_generator=test_datagen.flow_from_directory(validation_dir,
                                                 target_size=(128,128),
                                                 batch_size=64,
                                                 class_mode='categorical')

test_generator=test_datagen.flow_from_directory(test_dir,
                                                 target_size=(128,128),
                                                 batch_size=64,
                                                 class_mode='categorical')



#########################################################################################
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import regularizers

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128,128, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(4, activation='softmax'))


from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical

model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(),metrics=["acc"])

# Print the model summary
model.summary()



##########################################################################################


# Train the model

history = model.fit(train_generator, epochs=50, batch_size=64, validation_data=validation_generator)

#############################################################
#testing the model

test_generator=test_datagen.flow_from_directory(test_dir,target_size=(128,128),batch_size=64,class_mode='categorical')
model.evaluate(test_generator)


#####################

output_folder = r'C:\Users\USER\Desktop\final result'   
os.makedirs(output_folder, exist_ok=True) 

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Define the test data directory
test_dir=r'C:\Users\USER\Desktop\testset' # Replace with the path to your test set folder

# Create an ImageDataGenerator for test data
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load the test data using flow_from_directory
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),  # Set the target image size as needed
    batch_size=64,
    class_mode='categorical',
    shuffle=False  # Set shuffle to False to keep the order of the files
)

# Make predictions on the test data
predicted_labels = model.predict(test_generator)

# Convert predicted labels to class indices (0, 1, 2, 3)
predicted_indices = np.argmax(predicted_labels, axis=1)

# Get the true labels from the test generator
true_labels = test_generator.classes

# Compute the confusion matrix
confusion = confusion_matrix(true_labels, predicted_indices)

# Define class names based on the subfolder names
class_names = list(test_generator.class_indices.keys())

# Calculate the confusion matrix as percentages
confusion_percent = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis] * 100

# Plot the confusion matrix as percentages
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'  # Format as percentages
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt) + '%', horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot the confusion matrix as percentages

plt.figure(figsize=(8, 6))
plot_confusion_matrix(confusion_percent, classes=class_names, title='Confusion matrix (in %)', cmap=plt.cm.Blues)
plt.savefig(os.path.join(output_folder, "confusion_matrix.001kernels.png"))  # Save the confusion matrix as a PNG image
plt.show()



# Save the graph as an image (e.g., in PNG format) in the output folder
plt.figure(figsize=(8, 5))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Training History")
graph_path = os.path.join(output_folder, "GRAPH_.001kernels.png")
plt.savefig(graph_path)
plt.close()

report = classification_report(true_labels, predicted_indices, target_names=class_names)

# Save the classification report as a text file
with open(os.path.join(output_folder, "classification_repor..001kernels.txt"), "w") as report_file:
    report_file.write("Classification Report:\n")
    report_file.write(report)

print(f"Results saved in {output_folder}")


#########################################STOP##########################################################

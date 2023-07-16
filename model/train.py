import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Defining the paths to the dataset
train_data='train'
test_data='test'

#defining the number of classes and other parameters 
num_classes = 6
input_shape= (64,64,3)
batch_size=32
epochs=15

#normalisation
train_data_gen=ImageDataGenerator(rescale=1./255,
                                  rotation_range=20,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
test_data_gen = ImageDataGenerator(rescale=1./255)

#Preprocess and augment the training images
train_generator = train_data_gen.flow_from_directory(
    train_data,
    target_size=(input_shape[0],input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load and preprocess the testing dataset
test_generator = test_data_gen.flow_from_directory(
    test_data,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical'
)


# Define the CNN model architecture with L2 regularization
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=num_classes, activation='softmax'))

#compile the model

model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])


#train the model
history=model.fit(train_generator,
                  steps_per_epoch=train_generator.n//batch_size,
                  epochs=epochs,validation_data= test_generator,
                  validation_steps=test_generator.samples//batch_size)

#save the model
model.save('notes_cnn_model.h5')


# Evaluate the model on the testing dataset
test_loss, test_accuracy = model.evaluate(test_generator)

# Compute the predicted labels for the testing dataset
predicted_labels = model.predict(test_generator)
predicted_classes = np.argmax(predicted_labels, axis=1)

# Get the true labels for the testing dataset
true_labels = test_generator.classes

# Compute the accuracy
accuracy = np.sum(predicted_classes == true_labels) / len(true_labels)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)
print('Computed Accuracy:', accuracy)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)


plt.plot(epochs, accuracy, 'bo-', label='Train')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


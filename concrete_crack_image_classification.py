#%%
#1. Setup  - import packages
import tensorflow as tf
from  tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks, applications, models
import numpy as np
import matplotlib.pyplot as plt
import os, datetime
import cv2
import imghdr

#%%
#2. Data loading
#_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
#path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname('C:\\Users\\ruzan\\Documents\\Ruzana\\SHRDC\\DL\\Hands_On\\dlcv\\transfer_learning\\Concrete_Crack_Prediction\\data'), 'data')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'test')

BATCH_SIZE = 32
IMG_SIZE = (227,227)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

#%%
#3. Inspect some data examples
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    plt.grid("off")
#%%
#4. Further split the validation dataset into validation-test splits
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 2)
validation_dataset = validation_dataset.skip(val_batches // 2)

#%%
#5. Convert tensoflow dataset into PerfetchDataset
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#%%
#6. Create a sequential model for image augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

#%%
#7. Visualizing data augmentation
for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')

#%%
#8. Data normalization - define a layer for it
preprocess_input = applications.mobilenet_v2.preprocess_input

#%%
#9. Construct the transfer learning pipeline
#Pipeline: data augmentation > preprocess input> transfer learning model
#(A) Load the pretrained model using keras.applications
# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.summary()
keras.utils. plot_model(base_model)
# %%
#(B) Freeze the entire feature extractor
base_model.trainable = False
base_model.summary()

#%%
#(C) Create golbal average pooling layer
global_avg = layers.GlobalAveragePooling2D()
#(D) Create the output layer with Dense layer
output_layer = layers.Dense(len(class_names), activation='softmax')
#(E) Build the entire pipeline using functional API 
#a. Input
inputs = keras.Input(shape=IMG_SHAPE)
#b. Data augemntation
x = data_augmentation(inputs)
#c. Data normalization
x = preprocess_input(x)
#d. Transfer learning feature extractor
x = base_model(x, training=False)
#e. Classification layers
x = global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)
#f. Build the model
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

#%%
#10. Compile the model
optimizer = optimizers.Adam(learning_rate = 0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

#%%
#Prepare the callback objects for model.fit()
early_stopping = callbacks.EarlyStopping(patience=2)
PATH = os.getcwd()
logpath = os.path.join(PATH,"tensorboard_log")
datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = callbacks.TensorBoard(logpath)

#%%
#Evaluate the model with test data before training
model.evaluate(test_dataset)

#%%
#11. Model training
EPOCHS =10
history = model.fit(
  train_dataset,
  validation_data=validation_dataset,
  epochs = EPOCHS, callbacks = [early_stopping, tb]
)
#%%
#Evaluate the model after training
model.evaluate(test_dataset)
#%%
 
#%%
#12. Model deployment
#(A) Retrieve a batch of data from test data and perform prediction
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)

#%%
#(B) Identify the class for the predictions
prediction_indexes = np.argmax(predictions,axis=1)
# %%
#(C) Display the result using matplotlib
label_map = {i:names for i, names in enumerate(class_names)}
prediction_list = [label_map[i] for i in prediction_indexes]
label_list = [label_map[i] for i in label_batch]
# %%
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(f"Prediction: {prediction_list[i]}, Label: {label_list[i]}")
    plt.axis("off")
    plt.grid("off")

#%% 
#13. Performance of the model
#Plot the loss graph
fig = plt.figure()
plt.plot(history.history['loss'], color='teal',label='loss')
plt.plot(history.history['val_loss'], color='orange',label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()
#%% 
#Plot the accuracy graph
fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()
# %%
#14. Save the model
from tensorflow.keras.models import load_model
model.save(os.path.join('models','classify_v2.h5'))
# %%

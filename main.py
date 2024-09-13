import os
import cv2
from PIL import Image
import numpy as np
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint


minmaxscaler = MinMaxScaler()

image_dataset = []
mask_dataset = []
image_patch_size = 256

for image_type in ['images', 'masks']:
    if image_type == 'images':
        image_ext = 'jpg'
    elif image_type == 'masks':
        image_ext = 'png'

    for tile_id in range(1, 8):
        for image_id in range(1, 10):
            # Read the image or mask
            image = cv2.imread(f'./Semantic segmentation dataset/Tile {tile_id}/{image_type}/image_part_00{image_id}.{image_ext}',1)
            if image is not None:
                if image_type == 'masks':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Resize the image to be divisible by patch size
                size_x = (image.shape[1] // image_patch_size) * image_patch_size
                size_y = (image.shape[0] // image_patch_size) * image_patch_size

                # Crop the image
                image = Image.fromarray(image)
                image = image.crop((0, 0, size_x, size_y))

                # Convert back to numpy array
                image = np.array(image)
                # Patchify the image
                patched_images = patchify(image, (image_patch_size, image_patch_size, 3), step=image_patch_size)

                for i in range(patched_images.shape[0]):
                    for j in range(patched_images.shape[1]):
                        if image_type == 'images':
                            individual_patched_image = patched_images[i, j, :, :]  # Accessing first index for RGB
                            individual_patched_image = minmaxscaler.fit_transform(
                                individual_patched_image.reshape(-1, individual_patched_image.shape[-1])).reshape(
                                individual_patched_image.shape)
                            individual_patched_image = individual_patched_image[0]
                            image_dataset.append(individual_patched_image)

                        elif image_type == 'masks':
                            individual_patched_mask = patched_images[i, j, :, :]  # Accessing first index for RGB
                            individual_patched_mask = individual_patched_mask[0]
                            mask_dataset.append(individual_patched_mask)

# print("Number of image patches:", len(image_dataset))
# print("Number of mask patches:", len(mask_dataset))
image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)
# plt.figure(figsize=(14, 8))
# plt.subplot(1, 2, 1)
# plt.imshow(image_dataset[0])
# plt.subplot(1, 2, 2)
# plt.imshow(mask_dataset[0])
# plt.show()

class_building = '#3C1098'
class_building = class_building.lstrip('#')
class_building = np.array(tuple(int(class_building[i:i + 2], 16) for i in (0, 2, 4)))
# print(class_building)

class_land = '#8429F6'
class_land = class_land.lstrip('#')
class_land = np.array(tuple(int(class_land[i:i + 2], 16) for i in (0, 2, 4)))
# print(class_land)

class_road = '#6EC1E4'
class_road = class_road.lstrip('#')
class_road = np.array(tuple(int(class_road[i:i + 2], 16) for i in (0, 2, 4)))
# print(class_road)

class_vegetation = '#FEDD3A'
class_vegetation = class_vegetation.lstrip('#')
class_vegetation = np.array(tuple(int(class_vegetation[i:i + 2], 16) for i in (0, 2, 4)))
# print(class_vegetation)

class_water = '#E2A929'
class_water = class_water.lstrip('#')
class_water = np.array(tuple(int(class_water[i:i + 2], 16) for i in (0, 2, 4)))
# print(class_water)

class_unlabeled = '#9B9B9B'
class_unlabeled = class_unlabeled.lstrip('#')
class_unlabeled = np.array(tuple(int(class_unlabeled[i:i + 2], 16) for i in (0, 2, 4)))
# print(class_unlabeled)

mask_dataset.shape[0]
label = individual_patched_mask

def rgb_to_label(label):
  label_segment = np.zeros(label.shape, dtype=np.uint8)
  label_segment[np.all(label == class_water, axis=-1)] = 0
  label_segment[np.all(label == class_land, axis=-1)] = 1
  label_segment[np.all(label == class_road, axis=-1)] = 2
  label_segment[np.all(label == class_building, axis=-1)] = 3
  label_segment[np.all(label == class_vegetation, axis=-1)] = 4
  label_segment[np.all(label == class_unlabeled, axis=-1)] = 5
  #print(label_segment)
  label_segment = label_segment[:,:,0]
  #print(label_segment)
  return label_segment

labels = []
for i in range(mask_dataset.shape[0]):
  label = rgb_to_label(mask_dataset[i])
  labels.append(label)

# print(len(labels))

labels=np.array(labels)
labels = np.expand_dims(labels, axis=3)
# print(labels[0])

np.unique(labels)
# print("Total unique labels based on masks: ",format(np.unique(labels)))

random_image_id = random.randint(0, len(image_dataset))

# plt.figure(figsize=(14,8))
# plt.subplot(121)
# plt.imshow(image_dataset[random_image_id])
# plt.subplot(122)
# plt.imshow(mask_dataset[random_image_id])
# plt.imshow(labels[random_image_id][:,:,0])

labels[0][:,:,0]

total_classes = len(np.unique(labels))
labels_categorical_dataset = to_categorical(labels, num_classes=total_classes)

master_training_dataset = image_dataset

X_train, X_test, y_train, y_test = train_test_split(master_training_dataset, labels_categorical_dataset, test_size=0.15,
                                                    random_state=100)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

image_height = X_train.shape[1]
image_width = X_train.shape[2]
image_channels = X_train.shape[3]
total_classes = y_train.shape[3]


# print(image_height)
# print(image_width)
# print(image_channels)
# print(total_classes)

from tensorflow.keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import concatenate, BatchNormalization, Dropout, Lambda

from keras import backend as K

def jaccard_coef(y_true, y_pred):
  y_true_flatten = K.flatten(y_true)
  y_pred_flatten = K.flatten(y_pred)
  intersection = K.sum(y_true_flatten * y_pred_flatten)
  final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
  return final_coef_value

def multi_unet_model(n_classes=5, image_height=256, image_width=256, image_channels=1):

  inputs = Input((image_height, image_width, image_channels))

  source_input = inputs

  c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(source_input)
  c1 = Dropout(0.2)(c1)
  c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
  p1 = MaxPooling2D((2,2))(c1)

  c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
  c2 = Dropout(0.2)(c2)
  c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
  p2 = MaxPooling2D((2,2))(c2)

  c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
  c3 = Dropout(0.2)(c3)
  c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
  p3 = MaxPooling2D((2,2))(c3)

  c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
  c4 = Dropout(0.2)(c4)
  c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
  p4 = MaxPooling2D((2,2))(c4)

  c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
  c5 = Dropout(0.2)(c5)
  c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)

  u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c5)
  u6 = concatenate([u6, c4])
  c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
  c6 = Dropout(0.2)(c6)
  c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

  u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(c6)
  u7 = concatenate([u7, c3])
  c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
  c7 = Dropout(0.2)(c7)
  c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

  u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding="same")(c7)
  u8 = concatenate([u8, c2])
  c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
  c8 = Dropout(0.2)(c8)
  c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

  u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding="same")(c8)
  u9 = concatenate([u9, c1], axis=3)
  c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
  c9 = Dropout(0.2)(c9)
  c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

  outputs = Conv2D(n_classes, (1,1), activation="softmax")(c9)

  model = Model(inputs=[inputs], outputs=[outputs])
  return model

metrics = ["accuracy", jaccard_coef]
print(image_height)
print(image_width)
print(image_channels)
print(total_classes)

def get_deep_learning_model():
    return multi_unet_model(n_classes=total_classes,
                            image_height=image_height,
                            image_width=image_width,
                            image_channels=image_channels)


model = get_deep_learning_model()

# Please uncomment this line to get model configuration
print(model.get_config())

weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]

import segmentation_models as sm

dice_loss = sm.losses.DiceLoss(class_weights=weights)

focal_loss = sm.losses.CategoricalFocalLoss()

total_loss = dice_loss + (1 * focal_loss)

# Define the checkpoint filepath
# checkpoint_filepath = 'model_checkpoint_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5'
#
# Create a ModelCheckpoint callback to save the model after every epoch or when the validation loss improves
# checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
#                                       save_best_only=True,
#                                       monitor='val_loss',
#                                       mode='min',
#                                       verbose=1)
#
# import tensorflow as tf
#
# tf.keras.backend.clear_session()
#
# model.compile(optimizer="adam", loss=total_loss, metrics=metrics)
#
# model.summary()
#
# model_history = model.fit(X_train, y_train,
#                           batch_size=16,
#                           verbose=1,
#                           epochs=100,
#                           validation_data=(X_test, y_test),
#                           shuffle=False,
#                           callbacks=[checkpoint_callback])
#
# history_a = model_history
# print(history_a.history)
#
# loss = history_a.history['loss']
# val_loss = history_a.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label="Training Loss")
# plt.plot(epochs, val_loss, 'r', label="Validation Loss")
# plt.title("Training Vs Validation Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
#
# jaccard_coef = history_a.history['jaccard_coef']
# val_jaccard_coef = history_a.history['val_jaccard_coef']
#
# epochs = range(1, len(jaccard_coef) + 1)
# plt.plot(epochs, jaccard_coef, 'y', label="Training IoU")
# plt.plot(epochs, val_jaccard_coef, 'r', label="Validation IoU")
# plt.title("Training Vs Validation IoU")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
#
#
# y_pred = model.predict(X_test)
# print(len(y_pred))
# print(y_pred)
# y_pred_argmax = np.argmax(y_pred, axis=3)
# print(len(y_pred_argmax))
# print(y_pred_argmax)
y_test_argmax = np.argmax(y_test, axis=3)
# print(y_test_argmax)
#
# model.save('unet_segmentation_model.h5')
import tensorflow as tf
model = tf.keras.models.load_model('unet_segmentation_model.h5',
                                   custom_objects={'dice_loss_plus_1focal_loss': None, 'jaccard_coef': None})
# pred = model.predict(X_train)
# plt.imshow(pred[0])

import random

test_image_number = random.randint(0, len(X_test))

test_image = X_test[test_image_number]
ground_truth_image = y_test_argmax[test_image_number]

test_image_input = np.expand_dims(test_image, 0)

prediction = model.predict(test_image_input)
predicted_image = np.argmax(prediction, axis=3)
predicted_image = predicted_image[0, :, :]

plt.figure(figsize=(14,8))
plt.subplot(231)
plt.title("Original Image")
plt.imshow(test_image)
plt.subplot(232)
plt.title("Original Masked image")
plt.imshow(ground_truth_image)
plt.subplot(233)
plt.title("Predicted Image")
plt.imshow(predicted_image)
plt.show()


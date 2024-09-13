import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import numpy as np

# Load the model and bypass the unknown custom loss and metric functions
model = tf.keras.models.load_model('unet_segmentation_model.h5',
                                   custom_objects={'dice_loss_plus_1focal_loss': None, 'jaccard_coef': None})

# Load and preprocess the image
image = cv2.imread('/Users/pratyushguni/PycharmProjects/MachineLearningPrac/AerialImagery/Semantic segmentation dataset/Tile 1/images/image_part_001.jpg')
image = cv2.resize(image, (256, 256))  # Adjust size based on your model's input
image = image / 255.0  # Normalize the image
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Predict
predic = model.predict(image)

# Display the prediction
plt.imshow(np.squeeze(predic), cmap='gray')  # Assuming the output is a grayscale mask
plt.show()

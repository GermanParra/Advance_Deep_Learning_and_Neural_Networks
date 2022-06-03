import numpy as np
from keras import utils
from keras.applications import resnet

# Load Keras' ResNet50 model that was pre-trained against the ImageNet database
model = resnet.ResNet50()

# Load the image file, resizing it to 224x224 pixels (required by this model)
img = utils.load_img("bay.jpg", target_size=(224, 224))

# Convert the image to a numpy array
x = utils.img_to_array(img)

# Add a forth dimension since Keras expects a list of images
x = np.expand_dims(x, axis=0)

# Scale the input image to the range used in the trained network
x = resnet.preprocess_input(x)

# Run the image through the deep neural network to make a prediction
predictions = model.predict(x)

# Look up the names of the predicted classes. Index zero is the results for the first image.
predicted_classes = resnet.decode_predictions(predictions, top=10)

print("This is an image of:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood))


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# CNN params
input_dims = (32, 32, 3)
dims = (input_dims[0], input_dims[1])
classes = 2  # Assuming binary classification

# Set the path to your dataset (train and test directories)
train_dir = 'Images/RoadSignsRaw/train'
test_dir = 'Images/RoadSignsRaw/test'

# ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalize pixel values to [0, 1]
    rotation_range=30,  # Random rotations
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,  # Random shearing
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill mode for new pixels
)

test_datagen = ImageDataGenerator(rescale=1. / 255)  # Only rescaling for the test set

# Load the training and test data using the flow_from_directory method
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=dims,  # Resize all images to 32x32 pixels (like CIFAR-10)
    batch_size=64,  # Use a batch size of 64 or another appropriate number
    class_mode='binary',  # 'binary' for binary classification
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=dims,  # Resize all images to 32x32 pixels (like CIFAR-10)
    batch_size=64,  # Use a batch size of 64 or another appropriate number
    class_mode='binary',  # 'binary' for binary classification
    shuffle=True
)

# Create the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_dims),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(classes, activation='softmax')  # Softmax for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # For binary classification, this can be used
              metrics=['accuracy'])

# Train the model using the generator
history = model.fit(
    train_generator,  # Use the generator to provide data for training
    epochs=200,
    validation_data=test_generator  # Use the test generator for validation
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f"Test accuracy: {test_acc:.2f}")

# Make predictions on the test set
predictions = model.predict(test_generator)

# Retrieve class labels
class_labels = {v: k for k, v in train_generator.class_indices.items()}

# Display some of the test images and their predicted labels
# Display some of the test images and their predicted labels
for i, (image_batch, label_batch) in enumerate(test_generator):  # Iterate through batches
    if i == 1:  # If you just want to look at the first batch, break after the first batch
        break

    # image_batch will be a batch of images, and label_batch will be their corresponding labels
    for j in range(len(image_batch)):  # Iterate through the images in the batch
        image = image_batch[j]  # Get the j-th image in the batch
        label = label_batch[j]  # Get the j-th label in the batch

        # Get the predicted class index and map it to the class label
        predicted_class_index = predictions[i * test_generator.batch_size + j].argmax()  # Adjust based on batch index
        predicted_class_label = class_labels[predicted_class_index]  # Map index to class name

        # Get the actual class label
        actual_class_index = label  # This should be 0 or 1 for binary classification
        actual_class_label = class_labels[actual_class_index]  # Map index to class name

        # Display the image
        plt.imshow(image)  # Show the image
        plt.title(f"Predicted: {predicted_class_label}, Actual: {actual_class_label}")
        plt.show()


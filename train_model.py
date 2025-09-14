import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import os

# Settings
img_size = (224, 224)
batch_size = 32
epochs = 10
train_dir = "./train"
valid_dir = "./valid"

# Data generators
train_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
                                rotation_range=10,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                zoom_range=0.1,
                                horizontal_flip=True)

valid_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

train_data = train_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
valid_data = valid_gen.flow_from_directory(valid_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

# Build model
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_data, epochs=epochs, validation_data=valid_data)

# Save model
model.save("efficientnet_b0_trained.keras", save_format="keras")

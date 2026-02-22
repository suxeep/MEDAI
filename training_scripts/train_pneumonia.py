import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5

train_dir =r"C:\\Users\\rsude\Desktop\\HAL\\MEDAI\DS\\PNE\\chest_xray\\train"
val_dir = r"C:\\Users\\rsude\Desktop\\HAL\\MEDAI\DS\\PNE\\chest_xray\\val"

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False,
                         weights='imagenet')

base_model.trainable = False
# Freeze first 100 layers only
for layer in base_model.layers[:100]:
    layer.trainable = False
    
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator,
          validation_data=val_generator,
          epochs=EPOCHS)

model.save("pneumonia_model.h5")

print("Model Saved Successfully!")
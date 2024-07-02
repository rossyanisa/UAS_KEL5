import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras import layers, models, optimizers, callbacks

# Path ke direktori utama
base_dir = 'bikerider-dataset(done)'

# Pengaturan untuk ImageDataGenerator
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 999

# Menggunakan ImageDataGenerator untuk preprocessing
datagen = ImageDataGenerator(
    validation_split=0.2,  # Menyisihkan 20% data untuk validasi
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Menyiapkan data train dan data validation
train_data = datagen.flow_from_directory(
    base_dir,
    class_mode='categorical',
    subset='training',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

valid_data = datagen.flow_from_directory(
    base_dir,
    class_mode='categorical',
    subset='validation',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

# Verifikasi jumlah kelas
num_classes = len(train_data.class_indices)
print(f"Number of classes: {num_classes}")

# Menggunakan model pre-trained DenseNet201
base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
base_model.trainable = False  # Membekukan lapisan DenseNet201

# Menambahkan lapisan di atas model pre-trained
densenet_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compiling model
densenet_model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Callback untuk Early Stopping dan Learning Rate Scheduler
early_stopping_cb = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001
)

# Training model DenseNet201
densenet_hist = densenet_model.fit(
    train_data,
    epochs=50,
    validation_data=valid_data,
    callbacks=[early_stopping_cb, reduce_lr_cb]
)

# Path untuk menyimpan model
save_dir = 'models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Simpan model
model_path = os.path.join(save_dir, 'helmet_detection_model_densenet201.h5')
densenet_model.save(model_path)

print(f"Model saved to: {model_path}")

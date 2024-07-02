import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

# Path ke direktori utama
base_dir = 'bikerider-dataset(done)'

# Periksa apakah direktori ada
if not os.path.exists(base_dir):
    raise ValueError(f"Direktori {base_dir} tidak ditemukan. Pastikan pathnya benar.")

# Tampilkan isi direktori
print("Isi direktori utama:")
print(os.listdir(base_dir))

# Menghitung jumlah gambar pada dataset
number_label = {}
total_files = 0

for sub_dir in os.listdir(base_dir):
    sub_dir_path = os.path.join(base_dir, sub_dir)
    if os.path.isdir(sub_dir_path):
        number_label[sub_dir] = 0
        for sub_sub_dir in os.listdir(sub_dir_path):
            sub_sub_dir_path = os.path.join(sub_dir_path, sub_sub_dir)
            if os.path.isdir(sub_sub_dir_path):
                counting = len([f for f in os.listdir(sub_sub_dir_path) if os.path.isfile(os.path.join(sub_sub_dir_path, f))])
                number_label[sub_dir] += counting
                total_files += counting

print("\nTotal Files: " + str(total_files))
print("\nJumlah Gambar Tiap Label:")
for label, count in number_label.items():
    print(f"{label}: {count}")

# Visualisasi jumlah gambar tiap kelas
plt.bar(number_label.keys(), number_label.values())
plt.title("Jumlah Gambar Tiap Label")
plt.xlabel('Label')
plt.ylabel('Jumlah Gambar')
plt.xticks(rotation=45)
plt.show()

# Menampilkan sampel gambar tiap kelas
img_each_class = 1
img_samples = {}
classes = list(number_label.keys())

for c in classes:
    sub_dir_path = os.path.join(base_dir, c)
    for sub_sub_dir in os.listdir(sub_dir_path):
        sub_sub_dir_path = os.path.join(sub_dir_path, sub_sub_dir)
        if os.path.isdir(sub_sub_dir_path):
            temp = os.listdir(sub_sub_dir_path)[:img_each_class]
            for item in temp:
                img_path = os.path.join(sub_sub_dir_path, item)
                img_samples[c] = img_path

for label, img_path in img_samples.items():
    img = mpimg.imread(img_path)
    plt.title(label)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Pengaturan untuk ImageDataGenerator
IMAGE_SIZE = (200, 200)
BATCH_SIZE = 32
SEED = 999

# Menggunakan ImageDataGenerator untuk preprocessing
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.2  # Menyisihkan 20% data untuk validasi
)

# Menyiapkan data train dan data validation
train_data = datagen.flow_from_directory(
    base_dir,
    class_mode='categorical',  # Perbaiki kesalahan ejaan di sini
    subset='training',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

valid_data = datagen.flow_from_directory(
    base_dir,
    class_mode='categorical',  # Perbaiki kesalahan ejaan di sini
    subset='validation',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

# Verifikasi jumlah kelas
num_classes = len(train_data.class_indices)
print(f"Number of classes: {num_classes}")

# Image Augmentation
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.Rescaling(1./255)
    ]
)

# Membuat arsitektur model CNN
cnn_model = tf.keras.models.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])

# Compiling model
cnn_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# Training model CNN
cnn_hist = cnn_model.fit(
    train_data,
    epochs=20,
    validation_data = valid_data
)

# Membuat plot akurasi model CNN
plt.figure(figsize=(10,4))
plt.plot(cnn_hist.history['accuracy'])
plt.plot(cnn_hist.history['val_accuracy'])
plt.title('CNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()

# Membuat plot loss model CNN
plt.figure(figsize=(10,4))
plt.plot(cnn_hist.history['loss'])
plt.plot(cnn_hist.history['val_loss'])
plt.title('CNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()

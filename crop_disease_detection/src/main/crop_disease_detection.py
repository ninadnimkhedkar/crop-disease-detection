import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from torchvision import transforms
from tqdm import tqdm
import streamlit as st

# Define paths
train_dir = 'path/to/train/dataset'
val_dir = 'path/to/validation/dataset'

# Load images from folder
def load_images_from_folder(folder):
    images = []
    labels = []
    plant_folders = os.listdir(folder)
    for plant in tqdm(plant_folders, desc="Loading folders"):
        plant_path = os.path.join(folder, plant)
        if os.path.isdir(plant_path):
            image_files = os.listdir(plant_path)
            for image_file in tqdm(image_files, desc=f"Loading images in {plant} folder", leave=False):
                try:
                    image_path = os.path.join(plant_path, image_file)
                    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
                    image = tf.keras.preprocessing.image.img_to_array(image)
                    images.append(image)
                    labels.append(plant)
                except PermissionError:
                    print(f"Permission denied for file: {image_path}")
                except Exception as e:
                    print(f"Error loading file {image_path}: {e}")
    return np.array(images), np.array(labels)

# Load train and validation datasets
train_images, train_labels = load_images_from_folder(train_dir)
val_images, val_labels = load_images_from_folder(val_dir)

# Display and save sample images
def display_and_save_sample_images(images, labels, num_samples=6, output_folder='output_images'):
    num_samples = min(num_samples, len(images))
    sample_indices = np.random.choice(len(images), num_samples, replace=False)
    sample_images = images[sample_indices]
    sample_labels = labels[sample_indices]
    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        plt.subplot(2, 3, i+1)
        image = sample_images[i].astype('uint8')
        plt.imshow(image)
        plt.title(sample_labels[i], y=-0.1)
        plt.axis('off')
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((100, 100))
        pil_image.save(os.path.join(output_folder, f'sample_image_{i+1}.png'))
    plt.tight_layout()
    plt.show()

# Display and save sample images from the training dataset
display_and_save_sample_images(train_images, train_labels)

# Normalize the images
train_images = train_images / 255.0
val_images = val_images / 255.0

# Filter unseen labels in validation set
train_labels_unique = set(train_labels)
val_labels_unique = set(val_labels)
unseen_labels = val_labels_unique - train_labels_unique
if unseen_labels:
    print(f"Unseen labels in validation set: {unseen_labels}")

val_images_filtered = []
val_labels_filtered = []
for image, label in zip(val_images, val_labels):
    if label not in unseen_labels:
        val_images_filtered.append(image)
        val_labels_filtered.append(label)

val_images_filtered = np.array(val_images_filtered)
val_labels_filtered = np.array(val_labels_filtered)

# Label encoding
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels_filtered)

# Convert labels to categorical format
train_labels_categorical = to_categorical(train_labels_encoded)
val_labels_categorical = to_categorical(val_labels_encoded)

# Build the CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

input_shape = (224, 224, 3)
num_classes = len(label_encoder.classes_)
cnn_model = create_cnn_model(input_shape, num_classes)

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(train_images, train_labels_categorical, epochs=20, batch_size=32, validation_data=(val_images_filtered, val_labels_categorical))

# Evaluate the CNN model
cnn_model.evaluate(val_images_filtered, val_labels_categorical)

# Load pre-trained Vision Transformer model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Load your dataset
def load_images_from_folder_vit(folder, num_images=1000):
    images = []
    count = 0
    for root, _, files in os.walk(folder):
        for filename in files:
            if count >= num_images:
                break
            img_path = os.path.join(root, filename)
            if img_path.endswith('.jpg') or img_path.endswith('.png'):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = np.array(img)
                    images.append(img)
                    count += 1
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue
    return images

train_images_vit = load_images_from_folder_vit('path/to/train/dataset', num_images=1000)
val_images_vit = load_images_from_folder_vit('path/to/validation/dataset', num_images=1000)

train_dataset = Dataset.from_dict({"image": train_images_vit})
val_dataset = Dataset.from_dict({"image": val_images_vit})
datasets = DatasetDict({"train": train_dataset, "validation": val_dataset})

# Preprocess the data
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def preprocess_data(dataset):
    for example in dataset:
        example['image'] = preprocess(example['image'])
    return dataset

train_dataset = preprocess_data(train_dataset)
val_dataset = preprocess_data(val_dataset)

# Fine-tuning the model
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
evaluation_results = trainer.evaluate()
print("Evaluation results:", evaluation_results)

# Streamlit application
st.title('Crop Classification and Disease Detection')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for CNN
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Crop classification
    predictions = cnn_model.predict(img_array)
    predicted_class = label_encoder.inverse_transform([np.argmax(predictions)])
    st.write(f"Predicted Crop Type: {predicted_class[0]}")

    # Preprocess the image for ViT
    img_array_vit = np.array(image.resize((256, 256)))
    img_array_vit = preprocess(img_array_vit)
    img_array_vit = img_array_vit.unsqueeze(0)

    # Disease detection
    with torch.no_grad():
        outputs = model(img_array_vit)
        predictions = outputs.logits.softmax(dim=-1).cpu().numpy()
        predicted_class = np.argmax(predictions, axis=-1)
        st.write(f"Predicted Disease: {predicted_class[0]}")

print("Streamlit app created.")

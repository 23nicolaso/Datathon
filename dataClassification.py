from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from tensorflow.keras import layers, models
import os


def train_model():
    # Load the dataset
    data = pd.read_csv('trainingData.csv')
    data['image_path'] = data.apply(lambda row: os.path.join('downloaded_images', f"{row.name}_{row['Address'].replace(' ', '_')}.jpg"), axis=1)

    # Check if the image files exist before proceeding
    data['image_exists'] = data['image_path'].apply(os.path.exists)
    data = data[data['image_exists']]

    # Ensure there is data to proceed with
    if data.empty:
        raise ValueError("No image files found. Ensure the 'downloaded_images' directory contains the images.")

    # Split the dataset
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Image data generator
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=None,
        x_col='image_path',
        y_col='AnsType',
        target_size=(256, 256),
        class_mode='categorical',
        batch_size=32
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_data,
        directory=None,
        x_col='image_path',
        y_col='AnsType',
        target_size=(256, 256),
        class_mode='categorical',
        batch_size=32
    )

    # Ensure there is data to proceed with for model training
    if train_generator.n == 0 or test_generator.n == 0:
        raise ValueError("No data available for training or testing. Check your data split or image paths.")

    # Build the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_generator, validation_data=test_generator, epochs=100)

    # Save the model
    model.save('my_model.hdf5')


# Define the model loading function outside of the prediction function to avoid repeated loading
model_cache = {}  # Cache to store loaded models
def load_model(model_path):
    if model_path not in model_cache:
        model_cache[model_path] = tf.keras.models.load_model(model_path)
    return model_cache[model_path]

def predict_image_classification(image_path, model_path='my_model.hdf5'):
    # Load the trained model
    model = load_model(model_path)
    
    # Load and prepare the image
    img = Image.open(image_path).resize((256, 256))
    img_array = np.array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    img_preprocessed = tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
    
    # Make a prediction
    prediction = model.predict(img_preprocessed)
    
    # Decode the prediction
    predicted_class = np.argmax(prediction, axis=1)
    
    # Assuming the class_indices dictionary is available globally or can be loaded
    # This part needs to be adapted based on how classes are handled in the training script
    class_indices = {'U': 0, 'G': 1, 'C': 2, 'F': 3, 'N': 4}  # Example class_indices
    classes = list(class_indices.keys())
    predicted_class_name = classes[predicted_class[0]]
    
    return predicted_class_name


# 1. Data Importation
def import_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
# 2. Satellite Imagery Acquisition
def download_satellite_image(api_key, lat, lon, zoom_level=18):
    base_url = "http://dev.virtualearth.net/REST/V1/Imagery/Map/Aerial"
    request_url = f"{base_url}/{lat},{lon}/{zoom_level}?\&key={api_key}&mapSize=256,256"
    print(request_url)
    try:
        response = requests.get(request_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
        elif response.status_code == 429:
            print("Your request could not be completed because of too many requests.")
            return None
        else:
            print("Failed to download image.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
# Main Loop
if __name__ == "__main__":
    file_path = "buildings.csv"
    df = import_data(file_path)
    
    api_key = "AsUyJHdv8-antEw81uskUKQM1m_lTry4wvDNNMH_E4Jfrvj6yVD5VS7YRUyBYvm6"
    
    images = []
    
    # Check if 'Latitude' and 'Longitude' columns exist before proceeding
    if df is not None and 'Latitude' in df.columns and 'Longitude' in df.columns:
        for index, row in df.iterrows():
            image = download_satellite_image(api_key, row['Latitude'], row['Longitude'])
            if image is not None:
                images.append(image)
    else:
        print("The DataFrame does not contain 'Latitude' and 'Longitude' columns or is empty.")
    
    # Create a directory for images if it doesn't exist
    images_folder = "downloaded_images"
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    # Export images to the folder with number and address as filename
    for i, (index, row) in enumerate(df.iterrows()):
        if i < len(images):
            image_path = os.path.join(images_folder, f"{i}_{row['Address'].replace(' ', '_')}.jpg")
            images[i].save(image_path)

    # Add a column to df where each image is classified using predict_image_classification
    classification_results = []
    for image_path in os.listdir(images_folder):
        full_image_path = os.path.join(images_folder, image_path)
        classification_result = predict_image_classification(full_image_path)
        classification_results.append(classification_result)
    
    # Assuming the images are processed in the same order as the rows in the DataFrame
    df['Classification'] = classification_results

    if df is not None:
        df.to_csv(file_path, index=False)
    else:
        print("No data to write to CSV.")


import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam


def load_data(csv_file, image_folder):
    df = pd.read_csv(csv_file, header=None)
    df.columns = ['label']

    df['filename'] = df.index + 1
    df['filename'] = df['filename'].apply(lambda x: os.path.join(image_folder, f"File{x}.jpg"))

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='label',
        target_size=(256, 256),
        batch_size=20,
        class_mode='categorical',
        color_mode='rgb')

    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filename',
        y_col='label',
        target_size=(256, 256),
        batch_size=20,
        class_mode='categorical',
        color_mode='rgb')
    
    return train_generator, validation_generator

def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_generator, validation_generator):
    train_steps = len(train_generator)
    val_steps = len(validation_generator)
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,  
        epochs=100, 
        validation_data=validation_generator,
        validation_steps=val_steps
    )
    return model, history

if __name__ == '__main__':
    csv_file = '/Users/edwardkang/Desktop/CS-2124/Datathon/labeling.csv'
    image_folder = '/Users/edwardkang/Desktop/CS-2124/Datathon/downloaded_images'
    
    train_generator, validation_generator = load_data(csv_file, image_folder)
    num_classes = len(train_generator.class_indices)
    model = build_model(num_classes)
    model, history = train_model(model, train_generator, validation_generator)
    model.save('my_model.hdf5')


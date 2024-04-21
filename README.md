# Construction Site Analysis Toolkit

This toolkit comprises a set of Python scripts designed to facilitate the analysis of construction sites through satellite imagery classification, estimation of construction completion times, and data management. It leverages machine learning, specifically convolutional neural networks (CNNs), for image classification, and provides tools for downloading satellite images, estimating project completion times, and handling data.

## Features

- **Image Classification**: Utilizes a CNN to classify construction site images into predefined categories based on their construction phase.
- **Completion Time Estimation**: Estimates the number of quarters required to complete construction projects based on their size and current phase.
- **Satellite Imagery Download**: Downloads satellite images of construction sites given their latitude and longitude.
- **Data Management**: Imports, processes, and exports data related to construction sites, including satellite images and project details.

## Scripts Overview

### 1. `dataClassification.py`

- **Purpose**: Trains a CNN model for the classification of construction site images and predicts the construction phase of given images.
- **Key Functions**:
  - `train_model()`: Trains the CNN model using images stored in a specified directory.
  - `predict_image_classification(image_path, model_path)`: Predicts the construction phase of an image.

### 2. `estimateProduction.py`

- **Purpose**: Estimates the completion time of construction projects based on their size and classification.
- **Key Functions**:
  - `estimate_completion_quarters(size_sf, classification)`: Estimates the number of quarters required for completion.
  - `load_and_classify_properties(csv_path)`: Loads property data from a CSV file and estimates completion times.

### 3. `datdownload.py`

- **Purpose**: Manages data importation, satellite imagery download, and preliminary setup for image classification.
- **Key Functions**:
  - `import_data(file_path)`: Imports data from a specified CSV file.
  - `download_satellite_image(api_key, lat, lon, zoom_level)`: Downloads satellite images from Bing Maps API.

## Getting Started

To use this toolkit, you will need Python 3.6 or later. Dependencies include TensorFlow, Keras, Pandas, NumPy, Requests, PIL, and optionally GeoPandas for geographical data handling.

1. **Clone the repository**:
git clone https://github.com/23nicolaso/Datathon.git

2. **Install required libraries**:
pip install -r requirements.txt

3. **Download satellite images and classify**:
python datdownload.py

4. **Train the model**:
- use python dataClassification.py
- or use our pretrained model:
https://drive.google.com/file/d/10DPN6k1sAbjsJBzoIhnBdNbivHt5cZne/view?usp=sharing 

6. **Estimate project completion**:
python estimateProduction.py


## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

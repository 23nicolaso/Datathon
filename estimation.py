import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

model_path = '/Users/edwardkang/Desktop/CS-2124/Datathon/my_model.hdf5'
model = tf.keras.models.load_model(model_path)

def load_and_predict(image_path, model):
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)
    class_labels = {0: 'C', 1: 'F', 2: 'G', 3: 'N', 4: 'U'}
    return class_labels[predicted_class_index[0]]

size_to_quarters = {
    (0, 99999): 2,
    (100000, 299999): 3,
    (300000, 599999): 4,
    (600000, 999999): 5,
    (1000000, float('inf')): 6,
}

stage_to_percent = {
    'U': 0.05,
    'G': 0.15,
    'C': 0.35,
    'F': 0.65,
    'N': 0.95
}

def size_to_quarter_mapping(size):
    for (min_size, max_size), quarters in size_to_quarters.items():
        if min_size < size <= max_size:
            return quarters
    return 0

def calculate_sqft_distribution(remaining_quarters, remaining_sqft, current_stage):
    stage_weights = {
        'U': 0.5,
        'G': 0.8,
        'C': 1.0,
        'F': 1.2,
        'N': 1.5
    }
    
    current_weight = stage_weights[current_stage]
    weights = np.linspace(current_weight, stage_weights['N'], int(remaining_quarters))
    normalized_weights = weights / weights.sum() * remaining_sqft
    
    return normalized_weights.tolist()

def estimate_completion(start, size, stage):
    total_quarters_needed = size_to_quarter_mapping(size)
    completed_percent = stage_to_percent[stage]
    remaining_quarters = total_quarters_needed * (1 - completed_percent)
    remaining_sqft = size * (1 - completed_percent)
    start_year, start_quarter = map(int, start.split('.'))
    start_date = datetime(start_year, (start_quarter - 1) * 3 + 1, 1)
    remaining_full_quarters = int(remaining_quarters)
    completion_date = start_date + relativedelta(months=+3 * remaining_full_quarters)
    estimated_quarter = ((completion_date.month - 1) // 3) + 1
    estimated_year = completion_date.year
    if remaining_quarters > remaining_full_quarters:
        completion_date = completion_date + relativedelta(months=3)
        estimated_quarter = ((completion_date.month - 1) // 3) + 1
        estimated_year = completion_date.year
    estimated_completion_date = f"{estimated_year}.{estimated_quarter}"
    sqft_distribution = calculate_sqft_distribution(remaining_quarters, remaining_sqft, stage)
    quarters_list = []
    current_date = start_date
    for sqft in sqft_distribution:
        quarter = ((current_date.month - 1) // 3) + 1
        year = current_date.year
        quarters_list.append((f"{year}.{quarter}", sqft))
        current_date += relativedelta(months=3)
    return estimated_completion_date, quarters_list

csv_file_path = '/Users/edwardkang/Desktop/CS-2124/Datathon/atlanta start dates and square feet - Sheet1.csv'
buildings_df = pd.read_csv(csv_file_path, header=None)
buildings_df.columns = ['SquareFeet', 'StartQuarter']

quarter_totals = {}

for i in range(len(buildings_df)):
    image_path = f'/Users/edwardkang/Desktop/CS-2124/Datathon/atlanta images/File{i+1}.jpg'
    predicted_stage = load_and_predict(image_path, model)
    start_quarter = str(buildings_df.iloc[i, 1])
    size_sq_ft = buildings_df.iloc[i, 0]
    estimated_completion_date, sqft_quarters_list = estimate_completion(start_quarter, size_sq_ft, predicted_stage)
    

    print(f"File{i+1}.jpg - Predicted Stage: {predicted_stage} - Estimated Completion Date: {estimated_completion_date}")
    for quarter, sq_ft in sqft_quarters_list:
        print(f" - {quarter}: {sq_ft:.2f} square feet")
        if quarter in quarter_totals:
            quarter_totals[quarter] += sq_ft
        else:
            quarter_totals[quarter] = sq_ft


print("\nSummary of Total Square Feet Completed Per Quarter:")
for quarter, total_sq_ft in sorted(quarter_totals.items()):
    print(f"{quarter}: {total_sq_ft:.2f} square feet")

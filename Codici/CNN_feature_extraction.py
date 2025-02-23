
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm

# Main input folder containing all plot subfolders
input_folder = 'Training_Dataset'

# Process the complete dataset folder with the provided Excel file
excel_filename = input_folder +'/Training_Dataset.xlsx'

# Load the EfficientNetB4 model pre-trained on ImageNet
feature_extractor_model = EfficientNetB4(weights='imagenet', include_top=False, pooling='avg')

feature_folder = 'Features'
os.makedirs(feature_folder, exist_ok=True)  # Create the folder if it doesn't exist


###############################
#MEAN FEATURE EXTRACTION
###############################

# Define a function to load and preprocess an image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(380, 380))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract features in batches
def extract_features_in_batches(image_paths, batch_size=32):
    num_images = len(image_paths)
    num_batches = (num_images + batch_size - 1) // batch_size
    features = []

    start_time = time.time()

    for i in tqdm(range(num_batches), desc="Extracting features"):
        batch_paths = image_paths[i*batch_size:(i+1)*batch_size]
        batch_images = [load_and_preprocess_image(path) for path in batch_paths]
        batch_images = np.array(batch_images)
        batch_features = feature_extractor_model.predict(batch_images)
        for feature in batch_features:
            features.append(feature.flatten())

    end_time = time.time()
    avg_time_per_image = (end_time - start_time) / num_images
    print(f'Feature extraction completed in {end_time - start_time:.2f} seconds, averaging {avg_time_per_image:.4f} seconds per image.')

    return np.array(features)

# Function to process the dataset with all plots in a single folder



# Function to process the dataset with all plots in a single folder
def process_complete_dataset(dataset_folder, excel_filename, batch_size=32):
    # Load the Excel file
    df = pd.read_excel(excel_filename)

    # Dictionary to store features by plot
    plot_features = {}

    # Iterate over each subfolder (which corresponds to a plot)
    for plot_name in os.listdir(dataset_folder):
        plot_folder_path = os.path.join(dataset_folder, plot_name)

        if not os.path.isdir(plot_folder_path):
            continue  # Skip if it's not a directory

        # Get the list of image paths
        image_paths = [os.path.join(plot_folder_path, img) for img in os.listdir(plot_folder_path)
                       if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        image_paths = [img for img in image_paths if any(os.path.basename(img).startswith(prefix) for prefix in ['S1_', 'S2_', 'S3_', 'S4_', 'S5_', 'S6_', 'S7_', 'S8_', 'S9_', 'S10_', 'S11_'])]

        # Extract features for all images in the plot using batching
        if image_paths:
            plot_features[plot_name] = np.mean(extract_features_in_batches(image_paths, batch_size=batch_size), axis=0)

    # Combine with yield and crop data
    plot_info = df.set_index('Plot')[['Yield', 'Crop']].to_dict('index')

    # Create a final dataset
    final_data = []
    for plot, features in plot_features.items():
        # Use the plot name directly to get the plot information
        plot_info_entry = plot_info.get(plot, None)
        if plot_info_entry is not None:
            final_data.append((plot, features, plot_info_entry['Yield'], plot_info_entry['Crop']))

    # Convert to a pandas DataFrame for easier handling
    final_df = pd.DataFrame(final_data, columns=['Plot', 'Features', 'Yield', 'Crop'])

    # Save the resulting DataFrame to a file
    output_path = os.path.join(feature_folder,
                               f"{os.path.basename(feature_folder)}_training_dataset.pkl")
    final_df.to_pickle(output_path)

    return final_df


final_df = process_complete_dataset(input_folder, excel_filename, batch_size=32)


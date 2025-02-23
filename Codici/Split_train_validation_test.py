import os
import pandas as pd
import pickle
import random

# Set dataset and feature folder paths
DATASET_FOLDER = 'Codici\Datasets'
FEATURE_FOLDER = 'Codici\Features'
PICKLE_FILE_PATH = os.path.join(FEATURE_FOLDER, 'Features_Training_Dataset.pkl')

# Load the pickle file
with open(PICKLE_FILE_PATH, 'rb') as f:
    pickle_data = pickle.load(f)

# Ensure 'Plot' column is string
pickle_data['Plot'] = pickle_data['Plot'].astype(str)

# Extract RST (Yield) and Crop information for original plots
plot_info = pickle_data.drop_duplicates(subset=['Plot'])[['Plot', 'Yield', 'Crop']]

# Identify unique original plots (without augmentation)
original_plots = list(set([plot for plot in pickle_data['Plot'] if '_original' in plot]))

# Define fixed test set (10% of original plots, same across iterations)
num_test_plots = int(len(original_plots) * 0.1)
test_plots = original_plots[:num_test_plots]  # First 10% as test set

# Function to filter dataset for train/val with all augmentations
def filter_data(data, plots, include_augmentations=True):
    if include_augmentations:
        return data[data['Plot'].str.split('_').str[0].isin([plot.split('_')[0] for plot in plots])]
    else:
        return data[data['Plot'].isin(plots)]  # Only original plots in the test set

# Process 100 iterations with different train-validation splits
for i in range(1, 101):
    i_str = str(i)
    OUTPUT_DATASET_FOLDER = os.path.join(DATASET_FOLDER, f'Dataset_{i}')
    
    # Create dataset folder if not exists
    os.makedirs(OUTPUT_DATASET_FOLDER, exist_ok=True)

    # Shuffle remaining original plots for each iteration
    remaining_plots = [plot for plot in original_plots if plot not in test_plots]
    random.shuffle(remaining_plots)

    # Split remaining plots into train (70%) and validation (20%)
    num_train_plots = int(len(remaining_plots) * 0.7)
    train_plots = remaining_plots[:num_train_plots]
    validation_plots = remaining_plots[num_train_plots:]  # Remaining 20%

    # Filter datasets
    train_data = filter_data(pickle_data, train_plots, include_augmentations=True)
    validation_data = filter_data(pickle_data, validation_plots, include_augmentations=True)
    test_data = filter_data(pickle_data, test_plots, include_augmentations=False)  # Only original

    # Save .pkl files
    train_data.to_pickle(os.path.join(OUTPUT_DATASET_FOLDER, 'train_features.pkl'))
    validation_data.to_pickle(os.path.join(OUTPUT_DATASET_FOLDER, 'validation_features.pkl'))
    test_data.to_pickle(os.path.join(OUTPUT_DATASET_FOLDER, 'test_features.pkl'))

    # Create Excel report (only original plots with Crop & RST)
    report_path = os.path.join(OUTPUT_DATASET_FOLDER, 'dataset_report.xlsx')
    with pd.ExcelWriter(report_path) as writer:
        plot_info[plot_info['Plot'].isin(train_plots)].to_excel(writer, sheet_name='Train', index=False)
        plot_info[plot_info['Plot'].isin(validation_plots)].to_excel(writer, sheet_name='Validation', index=False)
        plot_info[plot_info['Plot'].isin(test_plots)].to_excel(writer, sheet_name='Test', index=False)

    print(f'Iteration {i}: Files saved to {OUTPUT_DATASET_FOLDER}')

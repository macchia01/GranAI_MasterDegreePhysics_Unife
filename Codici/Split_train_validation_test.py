import os
import pandas as pd
import pickle
import shutil


for i in range (1, 101):

    # Set dataset and output folder paths
    DATASET_FOLDER = 'Datasets'
    i = str(i)
    OUTPUT_DATASET_FOLDER = DATASET_FOLDER + f'/Dataset_{i}'
    # Create the output folder if it doesn't exist
    if not os.path.exists(OUTPUT_DATASET_FOLDER):
        os.makedirs(OUTPUT_DATASET_FOLDER)

    feature_folder = 'Features'
    # File paths
    pickle_file_path = feature_folder + '/Features_training_dataset.pkl'
    excel_file_path = feature_folder + f'/dataset_{i}' + '.xlsx'

    # Load the pickle file
    with open(pickle_file_path, 'rb') as f:
        pickle_data = pickle.load(f)

    # Load the Excel sheets and extract plot lists
    train_plots = pd.read_excel(excel_file_path, sheet_name='Train')['Plot'].tolist()
    validation_plots = pd.read_excel(excel_file_path, sheet_name='Validation')['Plot'].tolist()
    test_plots = pd.read_excel(excel_file_path, sheet_name='Test')['Plot'].tolist()

    # Ensure 'Plot' column in the pickle data is an integer where applicable
    pickle_data['Plot'] = pickle_data['Plot'].astype(str)  # Convert to string to match prefix searches

    # Function to filter data based on plot names including prefixes
    def filter_data(data, plots):
        filtered_data = pd.DataFrame()
        for plot in plots:
            # Add all rows where 'Plot' starts with the plot name followed by an underscore
            filtered_rows = data[data['Plot'].str.startswith(f"{plot}")]
            filtered_data = pd.concat([filtered_data, filtered_rows], ignore_index=True)
        return filtered_data

    # Filter the pickle data based on the plot names, including augmentations
    train_data = filter_data(pickle_data, train_plots)
    validation_data = filter_data(pickle_data, validation_plots)
    test_data = filter_data(pickle_data, test_plots)

    # Save the filtered data into new pickle files inside the specified folder
    train_data.to_pickle(os.path.join(OUTPUT_DATASET_FOLDER, 'train_features.pkl'))
    validation_data.to_pickle(os.path.join(OUTPUT_DATASET_FOLDER, 'validation_features.pkl'))
    test_data.to_pickle(os.path.join(OUTPUT_DATASET_FOLDER, 'test_features.pkl'))

    # Copy the Excel report to the new folder
    shutil.copy(excel_file_path, os.path.join(OUTPUT_DATASET_FOLDER, 'dataset_report.xlsx'))

    print(f'Files saved to {OUTPUT_DATASET_FOLDER}')

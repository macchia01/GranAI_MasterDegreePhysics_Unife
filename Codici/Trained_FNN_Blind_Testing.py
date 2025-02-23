###########################################
# LIBRARIES
###########################################

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import time

for i in range(1, 101):

    ###########################################
    # IMPORTS AND PREPARING TEST DATASET
    ###########################################

    i=str(i)

    BATCH_SIZE = 16

    # Nome del file .pkl da caricare (specifica il nome del file)
    pkl_filename = f'Features_Blind_Test.pkl'
    #pkl_filename = f'Allenamento_plot_aggregated_features_mean_augmentation_S5S8.pkl'
    dataset_dir = f'Dataset_{i}'
    # Directory settings
    test_dir = 'Features'  # Directory containing the test features pkl file
    #test_dir = 'Features'  # Directory containing the test features pkl file
    model_dir = f'Datasets/Training_' + dataset_dir # Directory containing the keras model
    model_name = 'modello_fcl_finale'  # Replace with your model's actual name
    output_dir = f'Datasets/Blind_Test/Blind_Test_' + dataset_dir # Directory for saving predictions (outside of Keras_Model)

    # Create the prediction directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the pre-trained model
    model = tf.keras.models.load_model(os.path.join(model_dir, f'{model_name}.keras'))

    # Function to load and prepare the test dataset
    def load_test_dataset(features, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(features)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    # Load test features (ignore 'Yield')
    test_df = pd.read_pickle(os.path.join(test_dir, pkl_filename))
    test_features = np.stack(test_df['Features'])

    # Prepare test dataset
    test_dataset = load_test_dataset(test_features, batch_size=BATCH_SIZE)

    # Calculate the number of steps for the test dataset
    test_steps = int(np.ceil(len(test_features) / BATCH_SIZE))


    ###########################################
    # PREDICTION
    ###########################################

    # Measure the time taken to make predictions
    start_time = time.time()
    predictions = model.predict(test_dataset, steps=test_steps, verbose=1)
    end_time = time.time()
    duration = end_time - start_time

    # Flatten predictions if necessary
    predictions = predictions.flatten()

    # Create a DataFrame for plot predictions
    plot_predictions_df = pd.DataFrame({
        'Plot': test_df['Plot'],  # Assuming Plot IDs are available in test_df
        'Crop': test_df['Crop'],  # Assuming Crop data is available in test_df
        'Predicted Yield': predictions
    })

    ###########################################
    # SAVE PREDICTIONS TO EXCEL
    ###########################################

    # Create a DataFrame for the test results (only duration in this case)
    test_results_df = pd.DataFrame({
        'Metric': ['Duration (s)'],
        'Value': [duration]
    })

    # Save the results to an Excel file with two sheets
    output_excel_path = os.path.join(output_dir, 'test_predictions.xlsx')
    with pd.ExcelWriter(output_excel_path) as writer:
        test_results_df.to_excel(writer, sheet_name='Test Results', index=False)
        plot_predictions_df.to_excel(writer, sheet_name='Plot Predictions', index=False)

    print(f"Results saved to {output_excel_path}")
    print(f"Prediction duration: {duration} seconds")


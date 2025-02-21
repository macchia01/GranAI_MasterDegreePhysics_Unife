
###########################################
#LIBRARIES
###########################################

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import skew
import time
from sklearn.metrics import r2_score


for i in range(1, 101):

    ###########################################
    #IMPORTS AND PREPARING TEST DATASET
    ###########################################

    i=str(i)

    BATCH_SIZE = 16

    # Carica il modello preaddestrato
    # Sostituisci il percorso con il tuo percorso reale dove è salvato il modello
    nome_dataset = f'Dataset_{i}'
    dataset_dir = f'Datasets/' + nome_dataset
    output_dir = f'Datasets/Training_' + nome_dataset
    test_dir = f'Datasets/Fixed_Test/Fixed_Test_' + nome_dataset
    os.makedirs(test_dir, exist_ok=True)
    model = tf.keras.models.load_model(output_dir + '/modello_fcl_finale.keras')

    def load_test_dataset(features, labels, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    test_df = pd.read_pickle(os.path.join(dataset_dir, 'test_features.pkl'))
    # Filter the test DataFrame to include only rows with "_original" in the Plot column
    test_df = test_df[test_df['Plot'].str.contains('_original', na=False)]

    test_features, test_labels = np.stack(test_df['Features']), test_df['Yield'].values
    test_labels = test_labels.reshape(-1, 1)

    # Assuming test_features and test_labels are already prepared
    test_dataset = load_test_dataset(test_features, test_labels, batch_size=BATCH_SIZE)

    # Calculate the number of steps for the test dataset
    test_steps = int(np.ceil(len(test_labels) / BATCH_SIZE))


    ###########################################
    #PREDICTION
    ###########################################

    # Measure the time taken to evaluate the model
    start_time = time.time()
    test_loss, test_mape = model.evaluate(test_dataset, steps=test_steps, verbose=1)
    end_time = time.time()
    duration = end_time - start_time

    print(f"Test Loss: {test_loss}")
    print(f"Test MAPE: {test_mape}")
    print(f"Evaluation Duration: {duration} seconds")

    # Make predictions on the test dataset
    predictions = model.predict(test_dataset, steps=test_steps, verbose=1)
    predictions = predictions.flatten()

    # Calculate MAPE for each individual plot
    mape_per_plot = np.abs((test_labels.flatten() - predictions) / test_labels.flatten()) * 100

    # Create a DataFrame for plot predictions
    plot_predictions_df = pd.DataFrame({
        'Plot': test_df['Plot'],  # Assuming Plot IDs are available in test_df
        'Crop': test_df['Crop'],  # Assuming Crop data is available in test_df
        'True Yield': test_labels.flatten(),
        'Predicted Yield': predictions,
        'MAPE (%)': mape_per_plot
    })

    # Calculate overall MAPE
    overall_mape = np.mean(mape_per_plot)

    # Add a final row for the overall MAPE
    final_row = pd.DataFrame({
        'Plot': ['Overall MAPE'],
        'Crop': [''],
        'True Yield': [''],
        'Predicted Yield': [''],
        'MAPE (%)': [''],
        'Overall MAPE (%)': [overall_mape]
    })

    # Append the final row to the DataFrame
    plot_predictions_df = pd.concat([plot_predictions_df, final_row], ignore_index=True)

    # Create a DataFrame for test results
    test_results_df = pd.DataFrame({
        'Metric': ['Test Loss', 'Test MAPE', 'Duration (s)'],
        'Value': [test_loss, test_mape, duration]
    })

    # Save the results to an Excel file with two sheets
    output_excel_path = test_dir + '/test_results.xlsx'
    with pd.ExcelWriter(output_excel_path) as writer:
        test_results_df.to_excel(writer, sheet_name='Test Results', index=False)
        plot_predictions_df.to_excel(writer, sheet_name='Plot Predictions', index=False)

    print(f"Results saved to {output_excel_path}")
    print(f"Overall MAPE: {overall_mape}")


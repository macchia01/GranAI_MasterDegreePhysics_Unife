
import pandas as pd
import os
import matplotlib.pyplot as plt


# Definisci la directory principale e la directory di output
# main_directory = 'Datasets/Fixed_Test'
main_directory = 'Datasets/Blind_Test'

# Define the output directory within the main directory
#output_directory = os.path.join(main_directory, f'Ensemble_Fixed_Test')
output_directory = os.path.join(main_directory, f'Ensemble_Blind_Test')

# Crea la directory di output se non esiste
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Dictionary to store data
data = {}

# Scan all subdirectories for Excel files
for root, dirs, files in os.walk(main_directory):
    for file in files:
        if file.endswith('.xlsx'):
            file_path = os.path.join(root, file)
            try:
                # Load the Excel file
                excel_data = pd.ExcelFile(file_path)

                # Check if 'Plot Predictions' sheet exists
                if 'Plot Predictions' in excel_data.sheet_names:
                    df = excel_data.parse('Plot Predictions')

                    # Ensure necessary columns exist
                    if 'Plot' in df.columns and 'Crop' in df.columns and 'Predicted Yield' in df.columns:
                        for _, row in df.iterrows():
                            key = (row['Plot'], row['Crop'])  # Unique key (Plot, Crop)

                            # Initialize entry if not exists
                            if key not in data:
                                data[key] = {'Predicted Yields': []}

                            # Append predicted yield values
                            data[key]['Predicted Yields'].append(row['Predicted Yield'])

                            # Handle 'True Yield' (only keep one instance if present)
                            if 'True Yield' in df.columns and not pd.isna(row['True Yield']):
                                data[key]['True Yield'] = row['True Yield']
                
                else:
                    print(f"Sheet 'Plot Predictions' not found in {file_path}")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

# Process collected data
final_data = []
for (plot, crop), values in data.items():
    predicted_yields = values['Predicted Yields']
    
    # Calculate Mean and Standard Deviation
    mean_predicted_yield = sum(predicted_yields) / len(predicted_yields) if predicted_yields else None
    std_dev_predicted_yield = (pd.Series(predicted_yields).std() if len(predicted_yields) > 1 else 0)

    # Build row dictionary
    row = {
        'Plot': plot,
        'Crop': crop,
        'Mean Predicted Yield': mean_predicted_yield,
        'Std Dev Predicted Yield': std_dev_predicted_yield
    }

    # Include 'True Yield' if it exists
    if 'True Yield' in values:
        row['True Yield'] = values['True Yield']

    final_data.append(row)

# Convert to DataFrame
df_final = pd.DataFrame(final_data)

# Define output file path
output_filename = os.path.basename(main_directory) + '_final_predictions.xlsx'
output_path = os.path.join(output_directory, output_filename)

# Save to Excel
df_final.to_excel(output_path, index=False)

print(f"File saved successfully at {output_path}")




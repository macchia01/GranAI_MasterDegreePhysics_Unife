
import pandas as pd
import os
import matplotlib.pyplot as plt


# Definisci la directory principale e la directory di output
main_directory = 'Datasets/Test'

# Define the output directory within the main directory
output_directory = os.path.join(main_directory, f'Ensemble_Test')

# Crea la directory di output se non esiste
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Lista per immagazzinare i dataframe
dfs = []

# Scorri tutte le sottocartelle per trovare i file Excel
for root, dirs, files in os.walk(main_directory):
    for file in files:
        if file.endswith('.xlsx'):
            file_path = os.path.join(root, file)
            try:
                # Carica il file Excel
                excel_data = pd.ExcelFile(file_path)

                # Verifica se esiste il foglio 'Plot Predictions'
                if 'Plot Predictions' in excel_data.sheet_names:
                    df = excel_data.parse('Plot Predictions')

                    # Rinomina la colonna 'Predicted Yield' per evitare conflitti
                    unique_identifier = len(dfs) + 1  # Può essere personalizzato se necessario
                    df.rename(columns={'Predicted Yield': f'Predicted Yield_{unique_identifier}'}, inplace=True)

                    # Aggiungi il dataframe alla lista
                    dfs.append(df)
                else:
                    print(f"Foglio non trovato in {file_path}")
            except Exception as e:
                print(f"Errore nel processare il file {file_path}: {e}")

# Procedi solo se sono stati trovati file validi
if dfs:
    # Unisci i dataframe su 'Plot' e 'Crop' con suffissi per evitare conflitti
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = pd.merge(df_merged, df, on=['Plot', 'Crop'], suffixes=('', f'_{len(dfs)}'))

    # Calcola la media e la deviazione standard per le colonne di resa
    yield_columns = [col for col in df_merged.columns if 'Predicted Yield' in col]
    df_merged['Mean Yield'] = df_merged[yield_columns].mean(axis=1)
    df_merged['Std Dev Yield'] = df_merged[yield_columns].std(axis=1)

    # Crea il nome del file di output basato sul nome della directory principale
    output_filename = os.path.basename(main_directory) + '_final_predictions.xlsx'

    # Salva il dataframe unito nella directory di output
    output_path = os.path.join(output_directory, output_filename)
    df_merged.to_excel(output_path, index=False)

    print(f"File salvato correttamente in {output_path}")

else:
    print("Nessun file valido con il foglio 'Plot Predictions' trovato.")



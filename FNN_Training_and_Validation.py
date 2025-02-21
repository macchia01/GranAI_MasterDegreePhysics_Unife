###########################################
#LIBRARIES
###########################################

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import callbacks
import time
from tensorflow.keras.applications import EfficientNetB4
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

for i in range (1,101):

    ###########################################
    #IMPORTS
    ###########################################

    i=str(i)
    nome_dataset = f'Dataset_{i}'
    # User parameters
    dataset_dir = 'Datasets/' + nome_dataset  # Directory where the .pkl files are located
    #output_dir = 'Output_' + dataset_dir  # Directory to save training results
    output_dir = 'Datasets/Training_' + nome_dataset
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    BATCH_SIZE = 32  # Batch size
    EPOCHS = 150  # Number of epochs for training
    SAVE_EVERY_N_EPOCHS = 1  # Frequency of model checkpoint saving
    EARLYSTOPPING_PATIENCE = 5  # Number of epochs with no improvement to stop training
    LEARNING_RATE = 0.001  # Initial learning rate
    LR_SCHEDULER_FACTOR = 0.5  # Factor to reduce learning rate
    LR_SCHEDULER_PATIENCE = 2  # Patience for learning rate reduction
    DROPOUT_FRACTION = 0.2  # Dropout rate
    DENSE_NEURONS = 32

    ###########################################
    #TENSORFLOW DATASET PREPARATION
    ###########################################

    # Load the aggregated features from .pkl files (assuming they are in the dataset_dir directly)
    train_df = pd.read_pickle(os.path.join(dataset_dir, 'train_features.pkl'))
    val_df = pd.read_pickle(os.path.join(dataset_dir, 'validation_features.pkl'))
    # Filter the test DataFrame to include only rows with "_original" in the Plot column
    #val_df = val_df[val_df['Plot'].str.contains('_original', na=False)]

    # Separate features and labels (yields)
    train_features, train_labels = np.stack(train_df['Features']), train_df['Yield'].values
    val_features, val_labels = np.stack(val_df['Features']), val_df['Yield'].values

    # Convert labels to the correct shape
    train_labels = train_labels.reshape(-1, 1)
    val_labels = val_labels.reshape(-1, 1)

    def load_feature_dataset(features, labels, batch_size, shuffle_buffer_size=1000):
        """
        Create a TensorFlow dataset from pre-extracted features and labels.

        Args:
            features (np.ndarray): Array of features (e.g., [n_samples, feature_dim]).
            labels (np.ndarray): Array of corresponding labels.
            batch_size (int): Batch size for training/validation.
            shuffle_buffer_size (int): Buffer size for shuffling the dataset.

        Returns:
            tf.data.Dataset: A TensorFlow dataset ready for training/validation.
        """
        # Create a TensorFlow dataset from the features and labels
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        # Shuffle, batch, and prefetch the dataset
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=False).repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    # Create the datasets
    train_dataset = load_feature_dataset(train_features, train_labels, batch_size=BATCH_SIZE)
    val_dataset = load_feature_dataset(val_features, val_labels, batch_size=BATCH_SIZE)


    ###########################################
    #FNN
    ###########################################

    # Assuming that `train_features` has been loaded and its shape is [num_samples, feature_dim]
    feature_dim = train_features.shape[1]
    # Define the input layer that matches the shape of the pre-extracted features
    inputs = tf.keras.Input(shape=(feature_dim,))
    x = layers.BatchNormalization()(inputs)
    x = layers.Dropout(DROPOUT_FRACTION)(x)
    x = layers.Dense(DENSE_NEURONS, activation='relu')(x)
    outputs = layers.Dense(1, name="pred")(x)
    # Create the final model
    model = tf.keras.Model(inputs, outputs)
    # Show the model summary
    model.summary()

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mean_squared_error',
        metrics=['mean_absolute_percentage_error']
    )

    ###########################################
    #CALLBACKS
    ###########################################

    # Callback per l'early stopping: interrompe l'addestramento se la validazione non migliora
    earlystopping = callbacks.EarlyStopping(
        monitor='val_loss',  # Monitora la perdita sulla validazione
        mode='min',  # Si ferma quando la perdita smette di diminuire
        patience=EARLYSTOPPING_PATIENCE,  # Numero di epoche senza miglioramento prima di interrompere
        restore_best_weights=True  # Ripristina i pesi del modello migliori raggiunti
    )

    # Callback per ridurre il learning rate se la validazione non migliora
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=LR_SCHEDULER_FACTOR,  # Fattore di riduzione del learning rate
        patience=LR_SCHEDULER_PATIENCE,  # Numero di epoche senza miglioramento prima di ridurre il learning rate
        verbose=1  # Stampa un messaggio quando il learning rate viene ridotto
    )


    # Calcolo del numero di passi per epoca (steps_per_epoch) e per la validazione (validation_steps)
    steps_per_epoch = len(train_labels) // BATCH_SIZE
    validation_steps = len(val_labels) // BATCH_SIZE


    # Callback personalizzato per salvare il modello ogni n epoche
    class CustomModelCheckpoint(callbacks.Callback):

        def __init__(self, filepath, save_freq, **kwargs):
            super().__init__(**kwargs)
            self.filepath = filepath
            self.save_freq = save_freq
            self.epochs_since_last_save = 0

        def on_epoch_end(self, epoch, logs=None):
            self.epochs_since_last_save += 1
            if self.epochs_since_last_save >= self.save_freq:
                self.epochs_since_last_save = 0
                filepath = self.filepath.format(epoch=epoch + 1)
                self.model.save(filepath, overwrite=True)  # Salva il modello
                print(f"Model saved at {filepath}")

    # Callback per salvare il modello ogni SAVE_EVERY_N_EPOCHS epoche
    checkpoint_callback = CustomModelCheckpoint(
        filepath=output_dir + '/modello_fcl_epoca_{epoch:02d}.keras',  # Percorso per salvare il modello
        save_freq=SAVE_EVERY_N_EPOCHS  # Frequenza di salvataggio in epoche
    )


    # Variabili per memorizzare informazioni durante l'addestramento
    epoch_durations = []
    train_mape = []
    train_losses = []
    val_mape = []
    val_losses = []
    time_per_image = []


    class EpochLogger(callbacks.Callback):

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()  # Inizia il timer per l'epoca

        def on_epoch_end(self, epoch, logs=None):
            epoch_duration = time.time() - self.epoch_start_time  # Calcola la durata dell'epoca
            epoch_durations.append(epoch_duration)

            # Logga le metriche di addestramento e validazione
            train_mape.append(logs.get('mean_absolute_percentage_error', 0) if logs else 0)
            train_losses.append(logs.get('loss', 0) if logs else 0)
            val_mape.append(logs.get('val_mean_absolute_percentage_error', 0) if logs else 0)
            val_losses.append(logs.get('val_loss', 0) if logs else 0)

            # Calcola il tempo per immagine durante l'epoca
            total_images = (steps_per_epoch * BATCH_SIZE) + (validation_steps * BATCH_SIZE)
            time_per_image_epoch = epoch_duration / total_images
            time_per_image.append(time_per_image_epoch)

        def on_train_end(self, logs=None):
            # Salva tutte le informazioni in un unico file Excel al termine dell'addestramento
            self.save_log()

        def save_log(self):
            # Salva le informazioni in un unico file Excel chiamato 'test_results.xlsx'
            df = pd.DataFrame({
                'Epoca': range(1, len(epoch_durations) + 1),
                'Durata [s]': epoch_durations,
                'Train MAPE': train_mape,
                'Train loss': train_losses,
                'Validation MAPE': val_mape,
                'Validation loss': val_losses,
                'Tempo per immagine [s]': time_per_image
            })
            with pd.ExcelWriter(output_dir + '/train_results.xlsx') as writer:
                df.to_excel(writer, sheet_name='Training Log', index=False)
                model_summary_df = self.get_model_summary_df()
                model_summary_df.to_excel(writer, sheet_name='Model Structure', index=False)
            print("Log unificato salvato in 'test_results.xlsx'")

        def get_model_summary_df(self):
            # Crea un DataFrame con il riepilogo della struttura del modello
            stringlist = []
            self.model.summary(print_fn=lambda x: stringlist.append(x))
            summary_str = "\n".join(stringlist)
            summary_list = summary_str.split("\n")
            df = pd.DataFrame(summary_list, columns=["Model Structure"])
            return df

    # Callback per loggare le informazioni sulle epoche
    epoch_logger = EpochLogger()


    # Callback per stampare le etichette dei batch all'inizio di ogni epoca
    class BatchLabelPrinter(callbacks.Callback):

        def __init__(self, dataset, num_batches=5):
            super(BatchLabelPrinter, self).__init__()
            self.dataset = dataset
            self.num_batches = num_batches

        def on_epoch_begin(self, epoch, logs=None):
            print(f"Epoch {epoch + 1} begins")
            # Stampa le etichette dei primi batch
            for batch_num, (images, labels) in enumerate(self.dataset.take(self.num_batches)):
                print(f"Batch {batch_num + 1}:")
                print(labels.numpy())
                print("-" * 50)


    # Istanzia il callback per stampare le etichette dei batch
    batch_label_printer = BatchLabelPrinter(train_dataset)



    ###########################################
    #TRAINING
    ###########################################

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1,
        callbacks=[epoch_logger, lr_scheduler, earlystopping]
    )



    ###########################################
    #LEARNING CURVES PLOT
    ###########################################
    # Updated function to plot training history with desired style
    def plot_training_history(history):
        # Extract epochs and metrics from the history object
        epochs = range(1, len(history.history['mean_absolute_percentage_error']) + 1)
        train_mape = history.history['mean_absolute_percentage_error']
        val_mape = history.history['val_mean_absolute_percentage_error']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Plot style parameters
        figsize = (10, 6)  # Figure size
        axis_label_fontsize = 16  # Font size for axis labels
        tick_label_fontsize = 16  # Font size for tick labels
        legend_fontsize = 16  # Font size for legend
        title_fontsize = 18  # Font size for the title

        # Plot MAPE (Mean Absolute Percentage Error)
        plt.figure(figsize=figsize)
        plt.plot(epochs, train_mape, marker='o', linestyle='-', label='Train MAPE', color='g')
        plt.plot(epochs, val_mape, marker='o', linestyle='-', label='Validation MAPE', color='r')
        plt.title('Model MAPE', fontsize=title_fontsize)
        plt.xlabel('Epoch', fontsize=axis_label_fontsize)
        plt.ylabel('MAPE [%]', fontsize=axis_label_fontsize)
        plt.legend(fontsize=legend_fontsize)
        plt.xticks(fontsize=tick_label_fontsize)
        plt.yticks(fontsize=tick_label_fontsize)
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.savefig(os.path.join(output_dir, 'mape_vs_epoch.png'), bbox_inches='tight')

        # Plot Loss
        plt.figure(figsize=figsize)
        plt.plot(epochs, train_loss, marker='o', linestyle='-', label='Train Loss', color='c')
        plt.plot(epochs, val_loss, marker='o', linestyle='-', label='Validation Loss', color='m')
        plt.title('Model Loss', fontsize=title_fontsize)
        plt.xlabel('Epoch', fontsize=axis_label_fontsize)
        plt.ylabel('Loss', fontsize=axis_label_fontsize)
        plt.legend(fontsize=legend_fontsize)
        plt.xticks(fontsize=tick_label_fontsize)
        plt.yticks(fontsize=tick_label_fontsize)
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.savefig(os.path.join(output_dir, 'loss_vs_epoch.png'), bbox_inches='tight')


    # Esegui la funzione per tracciare i grafici
    plot_training_history(history)


    ###########################################
    #SAVING MODEL
    ###########################################

    # Save the final .keras model
    model.save(output_dir + '/modello_fcl_finale.keras')


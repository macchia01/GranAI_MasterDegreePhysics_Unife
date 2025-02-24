# GranAI
*Implementing Deep Neural Networks for in situ crop yield prediction*

Questo repository contiene il progetto **GranAI**, incentrato sull’utilizzo di **Deep Neural Networks** e di **immagini RGB da UAV** per la stima della resa del frumento (Duro e Tenero) in campo. Il lavoro è basato sulla tesi di laurea magistrale dal titolo *"Implementing Deep Neural Networks for in situ crop yield prediction"*, svolta presso [Laboratory for Nuclear Technologies Applied to the Environment](https://www.fe.infn.it/radioactivity/) in collaborazione con Società Sementi SPA.

## Indice
- [Contesto e Obiettivi](#contesto-e-obiettivi)
- [Dataset e Struttura delle Cartelle](#dataset-e-struttura-delle-cartelle)
  - [Immagini dei Plot](#immagini-dei-plot)
  - [Feature Extraction con EfficientNetB4](#feature-extraction-con-efficientnetb4)
  - [File di Output](#file-di-output)
- [Suddivisione in Train, Validation e Test](#suddivisione-in-train-validation-e-test)
- [Addestramento con FNN](#addestramento-con-fnn)
- [Test del Modello](#test-del-modello)
- [Ensemble Predictions](#ensemble-predictions)
- [Requisiti e Setup](#requisiti-e-setup)
- [Come Citare il Lavoro](#come-citare-il-lavoro)
- [Contatti](#contatti)

---

## Contesto e Obiettivi
Il progetto **GranAI** nasce con l’obiettivo di sviluppare un sistema di *yield prediction* ad alta accuratezza per il frumento, sfruttando:
- Immagini RGB acquisite da drone (UAV) a bassa quota, per catturare caratteristiche fenotipiche legate alla resa.
- **EfficientNetB4** come CNN pre-addestrata per l’estrazione delle feature.
- Una **Feedforward Neural Network (FNN)** personalizzata, responsabile della regressione finale sulla resa.

La pipeline combina tecniche di *image processing*, *machine learning* e *transfer learning*, fornendo un framework completo dalla raccolta dati sul campo fino all’aggregazione finale delle predizioni (*Ensemble*) per una stima robusta della resa.

---

## Dataset e Struttura delle Cartelle

### Immagini dei Plot
Il dataset comprende **914 plot** di frumento (sia Duro sia Tenero), con **augmentazioni** applicate a ciascun plot. Le immagini sono organizzate nella cartella principale `Training_Dataset/`:

- Ogni **plot** presenta 5 versioni:
  - `_original`
  - `_flip_hor`
  - `_flip_vert`
  - `_rotate_90`
  - `_rotate_-90`
- In ciascuna di queste cartelle sono presenti **77 immagini** corrispondenti a 7 sezioni spaziali × 11 survey temporali (S1-S11).

```
Training_Dataset/
│── 1033_original/
│    ├── S1_C96_1033_0.jpg
│    ├── S1_C96_1033_1.jpg
│    ├── ...
│    ├── S11_C96_1033_6.jpg
│── 1033_flip_hor/
│── 1033_flip_vert/
│── 1033_rotate_90/
│── 1033_rotate_-90/
│── ...
```

Accanto alle immagini è presente anche un file Excel `Training_Dataset.xlsx`, con i **metadati** di ciascun plot.

### Feature Extraction con EfficientNetB4
Per ciascuna immagine (380×380 px), il modello **EfficientNetB4** (pre-addestrato su ImageNet) estrae un vettore di **1792 feature**. Avendo 77 immagini per ciascun plot + augmentazione, queste feature vengono **mediate** per ottenere un unico vettore rappresentativo per ogni combinazione *plot + augmentazione*.

Le feature estratte vengono salvate in formato Pickle (`.pkl`) all’interno di `Features/`.

#### File di Output
```
Features/
├── Features_training_dataset.pkl
```

---

## Suddivisione in Train, Validation e Test
La suddivisione in train, validation e test viene effettuata 100 volte per garantire robustezza:
- **Train** (70%)
- **Validation** (20%)
- **Test** (10%) (solo plot *original*)

Il codice `Split_train_validation_test.py` genera i dataset:
```
Datasets/
│── Dataset_X/
│    ├── train_features.pkl
│    ├── validation_features.pkl
│    ├── test_features.pkl
│    ├── dataset_report.xlsx
...
```

---

## Addestramento con FNN
`FNN_Training_and_Validation.py`:
- Carica i file `.pkl` di train e validation.
- Costruisce una **Feedforward Neural Network (FNN)**.
- Allena il modello e salva i risultati:
```
Datasets/
│── Training_Dataset_X/
│    ├── modello_fcl_finale.keras
│    ├── train_results.xlsx
│    ├── mape_vs_epoch.png
│    ├── loss_vs_epoch.png
```

---

## Test del Modello
Lo script `Trained_FNN_Testing.py` esegue il test del modello, calcolando:
- **MAPE** e **loss** su `test_features.pkl`.
- Salva i risultati in `test_results.xlsx`.

---

## Ensemble Predictions
`Ensemble_Predictions.py` aggrega le predizioni da diverse esecuzioni per una stima più robusta:
```
Datasets/
│── Fixed_Test/
│    ├── Ensemble_Test/
│        ├── Fixed_Test_final_predictions.xlsx
```

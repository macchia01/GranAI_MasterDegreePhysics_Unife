# 📁 Struttura del Dataset e Feature Extraction

## 📂 **Struttura delle Cartelle e File di Input**

Il dataset contiene immagini di **914 plot** di frumento (Duro e Tenero), con **augmentazioni** applicate a ciascun plot. Le immagini sono organizzate come segue:

### **📁 Training\_Dataset/**

📌 Contiene 914 plot con le seguenti augmentazioni:

- `_original`
- `_flip_hor`
- `_flip_vert`
- `_rotate_90`
- `_rotate_-90`

Ogni cartella `PLOT_AUGMENTATION` contiene **77 immagini** (7 sezioni spaziali × 11 survey temporali):

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

### **📊 Excel File (****`Training_Dataset.xlsx`****)**

Contiene metadati per ciascun plot:

| Plot             | RST (q/ha) | Crop          |
| ---------------- | ---------- | ------------- |
| 1033\_original   | 52.22      | Frumento Duro |
| 1033\_flip\_hor  | 52.22      | Frumento Duro |
| 1033\_flip\_vert | 52.22      | Frumento Duro |
| 1033\_rotate\_90 | 52.22      | Frumento Duro |
| ...              | ...        | ...           |

---

## 🏗 **Feature Extraction con EfficientNetB4**

- Il modello **EfficientNetB4** (preaddestrato su ImageNet) estrae vettori di **1792 feature** da ciascuna immagine.
- Per ogni **plot e augmentazione**, vengono estratti **77 vettori** (uno per immagine), che vengono **mediati** per ottenere un unico vettore rappresentativo.

### **⚙️ Opzioni nel Codice**

- **Includere immagini da tutti i survey** oppure **filtrarli**.
- **Batch processing** per ottimizzare l'estrazione delle feature.

---

## 📂 Struttura dell'Output

Le feature estratte vengono salvate in formato Pickle (`.pkl`).

### **📁 Features/**

```
Features/
│── Features_training_dataset.pkl
```

### **📄 Contenuto di ****`Features_training_dataset.pkl`**

| Plot | Feature Vector (1792-D)  | RST   | Crop            |
| ---- | ------------------------ | ----- | --------------- |
| 1033 | [-0.02, 0.13, ..., 0.87] | 52.22 | Frumento Duro   |
| 1034 | [0.05, -0.07, ..., 0.92] | 73.88 | Frumento Tenero |
| ...  | ...                      | ...   | ...             |

**➡️ Questo file viene poi usato per addestrare il modello FNN di predizione della resa.**




# 📁 Suddivisione dei Plot in Train, Validation e Test

## 📂 **Struttura delle Cartelle e File di Input**

Il codice prende i **914 plot** e li suddivide in:

- **Train (70%)**
- **Validation (20%)**
- **Test (10%)**

Questa suddivisione avviene **100 volte**, con una nuova distribuzione per ogni iterazione. La partizione viene fatta seguendo queste regole:

- Il **test set** include **solo le versioni originali** dei plot.
- Le **versioni augmentate** di ciascun plot sono assegnate solo a **train e validation**.
- **Train e Validation vengono reshufflati ad ogni iterazione**, mentre il **Test set rimane fisso**.
- La selezione dei plot per il test è **random**, così come la ridistribuzione dei rimanenti in train e validation.

---

## 📂 **Struttura delle Cartelle e File di Input**

### **📁 Features/** *(Cartella contenente i dati delle feature estratte)*
```
Features/
│── Features_training_dataset.pkl  # Pickle file con tutte le feature estratte
```

---

## 📂 **Struttura delle Cartelle e File di Output**
Per ogni iterazione (da 1 a 100), viene creata una cartella contenente i file filtrati:

### **📁 Datasets/** *(Contiene i sottoinsiemi generati)*
```
Datasets/
│── Dataset_1/
│    ├── train_features.pkl
│    ├── validation_features.pkl
│    ├── test_features.pkl
│    ├── dataset_report.xlsx
│── Dataset_2/
│── ...
│── Dataset_100/
```

### **📄 Struttura Interna dei File Pickle**
| Plot             | Feature Vector (1792-D) | RST   | Crop   |
| ---------------- | ----------------------- | ----- | ------ |
| 1033_flip_hor  | [-0.05, 0.17, ...]      | 52.22 | Duro   |
| 1033_flip_vert | [0.06, -0.03, ...]      | 52.22 | Duro   |
| 1034_original   | [0.12, 0.05, ...]       | 73.88 | Tenero |
| ...              | ...                     | ...   | ...    |

---

## 📄 **Struttura Interna di `dataset_report.xlsx`**
L'Excel di output contiene **tre fogli separati** per **Train, Validation e Test**, ognuno con la seguente struttura:

#### **📑 Train Sheet**
| Plot          | Crop         | RST (q/ha) |
|--------------|-------------|------------|
| 1033_flip_hor | Frumento Duro | 52.22 |
| 1033_flip_vert | Frumento Duro | 52.22 |
| 1034_rotate_90 | Frumento Tenero | 73.88 |
| ...          | ...         | ... |

#### **📑 Validation Sheet**
| Plot          | Crop         | RST (q/ha) |
|--------------|-------------|------------|
| 1034_original | Frumento Tenero | 73.88 |
| 1035_flip_hor | Frumento Duro | 65.55 |
| ...          | ...         | ... |

#### **📑 Test Sheet**
| Plot          | Crop         | RST (q/ha) |
|--------------|-------------|------------|
| 1035_original | Frumento Duro | 65.55 |
| 1036_original | Frumento Tenero | 78.12 |
| ...          | ...         | ... |

---



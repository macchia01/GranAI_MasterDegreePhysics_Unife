# Implementing-Deep-Neural-Networks-for-in-situ-crop-yield-prediction 🌾

**Tesi di Laurea Magistrale in Fisica** 
**Titolo:** Implementing Deep Neural Networks for in situ crop yield prediction  
**Candidato:** Nicola Macchioni  
**Supervisore:** Prof. Fabio Mantovani  
**Anno Accademico:** 2023/2024 (Università degli Studi di Ferrara)

## 📖 Descrizione del Progetto

Questo progetto implementa una pipeline basata su **Deep Neural Networks (DNN)** per la predizione della resa del grano utilizzando immagini RGB acquisite da **UAV (droni)**.  
Il modello sfrutta una **CNN pre-addestrata (EfficientNetB4)** per l'estrazione delle feature, seguita da un **Feedforward Neural Network (FNN)** per la regressione.

# **Implementing Deep Neural Networks for in situ crop yield prediction 🌾**

![Pipeline del Modello](images/image.png)

### 📜 Contenuto della Repository

- **[Tesi_NM.pdf](Tesi_NM.pdf)** → Tesi completa, con dettagli su dataset, metodologia e risultati.
- **Codici principali:**
  - `Split_train_validation_test.py` → Suddivide il dataset in train, validation e test set **(100 split diversi)**.
  - `CNN_feature_extraction.py` → Utilizza **EfficientNetB4** per estrarre feature dalle immagini.
  - `FNN_Training_and_Validation.py` → Addestra e valida il modello **Feedforward Neural Network**.
  - `Trained_FNN_Testing.py` → Testa il modello FNN con il dataset di test.
  - `Ensemble_Predictions.py` → Combina predizioni di più modelli per migliorare l'accuratezza.

# Implementing Deep Neural Networks for in situ crop yield prediction 🌾

<div style="display: flex; align-items: center;">
    <img src="images/image.png" alt="Pipeline del Modello" width="300" style="margin-right: 20px;">
    <div>
        <h2>Tesi di Laurea Magistrale in Fisica</h2>
        <p><b>Titolo:</b> Implementing Deep Neural Networks for in situ crop yield prediction</p>
        <p><b>Candidato:</b> Nicola Macchioni</p>
        <p><b>Supervisore:</b> Prof. Fabio Mantovani</p>
        <p><b>Anno Accademico:</b> 2023/2024 (Università degli Studi di Ferrara)</p>
    </div>
</div>

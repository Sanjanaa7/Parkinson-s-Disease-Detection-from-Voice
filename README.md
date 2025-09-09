# 🗣️ Parkinson’s Disease Detection from Voice

## 📌 Project Overview
Parkinson’s Disease (PD) is a progressive neurodegenerative disorder that affects movement, speech, and quality of life.  
This project leverages **Deep Learning** techniques to detect early signs of Parkinson’s Disease using **voice recordings**.  
Since voice changes are subtle but measurable, AI models can capture these patterns for **non-invasive and affordable screening**.

---

## 🎯 Goals
- Detect early Parkinson’s Disease from speech data.
- Use **voice features (jitter, shimmer, pitch)** or **raw audio (spectrograms/MFCCs)**.
- Build a **deep learning model (CNN / 1D-CNN / LSTM)** for classification.
- Provide interpretable predictions with Grad-CAM / SHAP.

---

## 📂 Dataset
- **UCI Parkinson’s Telemonitoring Voice Dataset**  
  🔗 [Download Link](https://archive.ics.uci.edu/ml/datasets/parkinsons)  

- Features: 22 biomedical voice measures per recording (jitter, shimmer, fundamental frequency, etc.)  
- Label: `status` (1 = Parkinson’s, 0 = Healthy)

Optional: Other datasets (PC-GITA, Oxford Parkinson’s) for raw audio experiments.

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Libraries:**  
  - `pandas`, `numpy`, `matplotlib`, `seaborn` (data processing & visualization)  
  - `scikit-learn` (preprocessing, evaluation)  
  - `tensorflow` / `keras` (deep learning models)  
  - `librosa` (audio processing, spectrograms, MFCCs)  

---

## 🚀 Workflow
1. **Data Collection** – Load dataset from UCI or other repositories.  
2. **Preprocessing**  
   - Normalize numeric features.  
   - For raw audio: extract MFCCs / spectrograms.  
3. **Modeling**  
   - DNN for tabular voice features.  
   - CNN / 1D-CNN + LSTM for raw audio.  
4. **Training & Evaluation**  
   - Metrics: Accuracy, F1-score, ROC-AUC.  
   - Handle class imbalance with class weights / SMOTE.  
5. **Explainability**  
   - Grad-CAM for spectrogram CNN.  
   - SHAP for feature importance.  
6. **Deployment (Future)**  
   - Simple **Streamlit web app** or **Flask API** for real-time voice-based screening.  

---

## 📊 Example Results (to be updated)
| Model        | Accuracy | Precision | Recall | F1-score |
|--------------|----------|-----------|--------|----------|
| DNN (tabular)| 89%      | 0.88      | 0.87   | 0.87     |
| CNN (MFCC)   | 92%      | 0.91      | 0.90   | 0.90     |


## 💡 Future Work
- Test on **larger multilingual voice datasets**.  
- Robustness testing under noisy environments.  
- Deploy as **mobile health app** for early Parkinson’s screening.  

---

## 👨‍💻 How to Run
```bash
# Clone repository
git clone https://github.com/your-username/parkinsons-voice-detection.git
cd parkinsons-voice-detection

# Install dependencies
pip install -r requirements.txt

# Run notebook in Google Colab
```
📜 License

This project is open-source under the MIT License.

🤝 Contributing

Pull requests and contributions are welcome!

🙌 Acknowledgements

UCI Parkinson’s Dataset

Librosa
 for audio feature extraction

TensorFlow
 for model building

 About Me

Name: Sanjanaa S

Course: B.Tech Artificial Intelligence and Data Science

College: Rajalakshmi Institute of Technology

Year: 3rd Year

Email: sanjanaasrinivasan7@gmail.com

LinkedIn: www.linkedin.com/in/sanjanaa-srinivasan-802ba5290

GitHub: https://github.com/Sanjanaa7



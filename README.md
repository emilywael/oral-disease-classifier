# 🦷 Oral Disease Classification using Deep Learning

An AI system that classifies oral disease images into 6 categories using a custom CNN and transfer learning (MobileNetV2, EfficientNetB0, ResNet50), deployed as an interactive web app.

## 🎯 Objective

Build, compare, and deploy a deep learning system capable of classifying oral disease images to support early screening and awareness.

## 📊 Dataset

- **Source:** [Oral Diseases Dataset](https://www.kaggle.com/datasets/salmansajid05/oral-diseases4) (Kaggle)
- **Classes (6):** Calculus, Data_Caries, Gingivitis, Hypodontia, Mouth_Ulcer, Tooth_Discoloration
- **Total images used:** 5,563 (original images only — augmented duplicates excluded to prevent data leakage)
- **Split:** 80% train / 10% validation / 10% test

## 🧠 Models & Results

| Model | Technique | Test Accuracy | Trainable Params |
|---|---|---|---|
| Custom CNN | Trained from scratch | 71.35% | 457,670 |
| MobileNetV2 | Transfer Learning + Fine-tuning | 78.00% | 2,587,462 |
| **EfficientNetB0** | **Transfer Learning (Feature Extraction)** | **80.07%** 🏆 | **4,379,049** |
| ResNet50 | Transfer Learning + Fine-tuning | 79.00% | 24,113,798 |

**EfficientNetB0** was selected as the final model — it achieved the highest accuracy with significantly fewer trainable parameters than ResNet50, offering the best accuracy-to-efficiency tradeoff.

### Key findings
- Fine-tuning improved MobileNetV2 and ResNet50, but *reduced* EfficientNetB0's performance — highlighting that fine-tuning sensitivity varies by architecture and isn't universally beneficial.
- Class imbalance (Gingivitis: 2,349 images vs. Tooth_Discoloration: 183 images) was addressed using computed class weights during training.
- Hyperparameter tuning (grid search over dense units, dropout rate, and learning rate) confirmed the original default configuration was near-optimal.

## 🚀 Deployment

The best-performing model (EfficientNetB0) is deployed as an interactive **Gradio** web application, allowing users to upload an oral image and receive real-time classification with confidence scores.

**🔗 Live Demo:** [Hugging Face Spaces link here]

### Run locally

```bash
git clone https://github.com/your-username/oral-disease-classifier.git
cd oral-disease-classifier
pip install -r requirements.txt
python app.py
```

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow / Keras
- **Models:** Custom CNN, MobileNetV2, EfficientNetB0, ResNet50 (ImageNet weights)
- **Deployment:** Gradio, Hugging Face Spaces
- **Data Processing:** NumPy, Pillow, split-folders

## 📁 Project Structure

```
oral-disease-classifier/
├── app.py                  # Gradio application
├── requirements.txt
├── model/
│   ├── final_model.keras   # Best model (EfficientNetB0)
│   └── class_names.json
├── notebook/
│   └── training_notebook.ipynb
└── README.md
```

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not a certified medical diagnostic tool and should not replace professional dental consultation.

## 📌 Future Improvements

- Expand dataset size for underrepresented classes (Data_Caries, Tooth_Discoloration)
- Experiment with ensemble methods combining multiple models
- Add Grad-CAM visualizations for model interpretability

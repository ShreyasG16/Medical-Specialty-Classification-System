# Medical Specialty Classification

## 📌 Project Description
This project focuses on building a **multi-class text classification system** to predict **40 medical specialties** from transcribed medical reports. The system processes medical transcription data, applies preprocessing, handles severe class imbalance, and evaluates multiple NLP models to determine the best-performing approach.

---

## 📊 Dataset Summary
- **Total Records:** 4,999 (expanded to 29,994 after augmentation)
- **Features:**
  - **description:** Brief reason for the medical visit
  - **medical_specialty:** Target label (40 unique classes)
  - **sample_name:** Case title
  - **transcription:** Full medical note text (main input)
  - **keywords:** List of related terms
- **Imbalance:** More than 90% class imbalance with Surgery class being dominating the most.
- **Augmentation:** Oversampling & rare-class merging increased minority representation to ~2% per class

---

## 🛠 Methodology
1. **Data Preprocessing:**
   - Text cleaning (lowercasing, removing special characters, extra whitespace)
   - Duplicate & missing value removal
   - Label normalization & rare-class merging
   - Feature engineering by combining description, transcription, and keywords

2. **Model Evaluation:**
   - **TF-IDF + SVM**
   - **Frozen BERT embeddings** with Logistic Regression / SVM
   - **Fine-tuned BERT**
   - Baseline comparison before and after augmentation

3. **Class Imbalance Handling:**
   - Oversampling minority classes via synonym augmentation technique.
   - Grouping rare classes into “Other” category
   - Augmentation increased dataset size by **500%**

---

## 📈 Results
| Model                  | Accuracy | Macro-F1 |
|------------------------|----------|----------|
| TF-IDF + SVM           | **82.8%** | **0.61** |
| Frozen BERT + Logistic | 72.1%    | 0.39     |
| Frozen BERT + SVM      | 70.9%    | 0.41     |
| Fine-tuned BERT        | 69.8%    | 0.37     |

- **Best Performer:** TF-IDF + SVM with augmentation
- Significant improvement in minority class recognition after augmentation

---

## 📂 Repository Structure
```
├── medical_specialty_classification.ipynb   # Jupyter Notebook with code
├── report.pdf                               # Detailed project report
└── README.md                                # Project documentation
```

---

## 🚀 Future Improvements
- Experiment with **ClinicalBERT** or **BioBERT** for domain-specific embeddings
- Implement advanced data augmentation for medical text
- Explore ensemble learning with both classical and transformer-based models

---

**Author:** Shreyas Gore  
**Tools:** Python, NLP, scikit-learn, Transformers, TF-IDF, SVM  

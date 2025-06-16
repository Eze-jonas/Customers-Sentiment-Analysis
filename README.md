 # 💬 Sentiment Analysis on Amazon Food Reviews

This project developed a predictive model using a deep learning algorithm (Convolutional Neural Network - CNN) from TensorFlow to analyze customer sentiments based on their product reviews. The goal is to leverage customer feedback to improve and maintain the quality of goods and services.

---

## 📊 Dataset

- **Name**: Amazon Fine Food Reviews  
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)  
- **Shape**: 568,454 rows × 7 columns  

---

## ⚙️ Model Choice Justification

To develop a sentence-level sentiment classifier, the **summary** column was selected.  
A **Convolutional Neural Network (CNN)** was used due to its efficiency and performance on short text. While advanced models like **LSTM** or **BERT** offer deeper understanding, their high resource demands made CNN the most practical and feasible choice, given RAM constraints and lack of a GPU.

---

## 🧹 Data Preprocessing

Preprocessing was performed in a **Databricks notebook** using a hybrid of **PySpark** and native **Python**.

- **PySpark**: Used for data manipulation and transformations
- **Python & Joblib**: Used for saving models and metadata outside the Spark context

### 🔠 Text Preprocessing Steps

- Trimmed extra spaces  
- Removed punctuation (except `!` and `?`, which signal sentiment)  
- Lowercased and tokenized text  
- Removed stopwords  
- Applied lemmatization  
- Reconstructed clean strings  
- Vectorized text using **Keras Tokenizer**  

### 🧾 Tokenizer Configuration

- Vocabulary size: 20,000 words  
- OOV token index: 1  
- Max sequence length: 100 (post-padding)  
- Word index broadcasted for PySpark compatibility  

---

## 🎯 Target Variable

- Target: **Score** column was preprocessed  
- Dropped irrelevant columns  

---

## 🔀 Dataset Splitting

- **Training**: 70%  
- **Validation**: 15%  
- **Testing**: 15%  
- **Cross-validation**: 5-fold  
- **Hyperparameter tuning**: Manual  

---

## 📈 Model Evaluation

| Dataset     | Accuracy | Precision | Recall | F1 Score | AUC    |
|-------------|----------|-----------|--------|----------|--------|
| Training    | 0.8849   | 0.8853    | 0.8849 | 0.8850   | 0.9781 |
| Validation  | 0.8681   | 0.8688    | 0.8681 | 0.8681   | 0.9702 |
| Test        | 0.8682   | 0.8690    | 0.8682 | 0.8683   | 0.9703 |

> 📌 These results indicate strong generalization and minimal overfitting.

---

## 🚀 Model Deployment (Databricks Interactive App)

The model was deployed within **Databricks** using **widgets** for interactivity. A simple form collects user inputs and runs the model end-to-end.

### Deployment Workflow:

1. Create new summary input using a widget form  
2. Load saved transformations and tokenizer  
3. Applied transformation manually to the new input  
4. Load trained model and make predictions  
5. Convert result from pandas → Spark DataFrame  
6. Merge with user input and store in **Delta Lake**  
7. Assign unique `PredictionID` for each entry  
8. Register Delta table in **SQL Catalog** for query access  

---

## 🧰 Tools Used

- **Languages & Libraries**: Python, PySpark, TensorFlow, Keras, joblib, SQL  
- **Platform**: Databricks  
- **Storage & Deployment**: Delta Lake, Databricks Widgets

---

## 📂 Files Included

| File | Description |
|------|-------------|
| `Sentiment_Analysis1.ipynb` | Dataset preprocessing and model development notebook |
| `Deployment Widgets.app2.ipynb` | widget-based model deployment notebook |
| `Sentiment_deployment_Widgets.docx` | Screenshots for the widget interface |

---

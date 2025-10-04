# **Parkinson's Disease Prediction Using Machine Learning Algorithms**

## **Authors**
- **Katsara Vasiliki**
- **Kontopoulos Christos**
- **Cavouras Dionisis** (Corresponding Author)  


**Department of Biomedical Engineering, University of West Attica, Greece**

---

## **Project Overview**

This project focuses on the **early detection of Parkinson's Disease (PD)** using **machine learning algorithms** to analyze **vocal signal data**. The goal is to evaluate different classifiers and feature-reduction techniques to identify the most effective methods for diagnosing PD.

---

## **Key Features**
- **Machine Learning Algorithms:** KNN, SVM, Random Forest, Logistic Regression, CART, LDA
- **Feature Reduction Methods:** PCA, RFE, Correlation Ranking, Statistical Significance Tests
- **Dataset:** Parkinson's voice dataset from Kaggle ([Link to dataset](https://www.kaggle.com/datasets/sagarbapodara/parkinson-csv))

---


## Results

The study showed that the **RFE** feature reduction method, in combination with either **Random Forest** or **KNN** classifiers, provided the best performance for detecting Parkinson's Disease (PD):  

- **Random Forest**:  
  - Accuracy: 94%  
  - Sensitivity: 87%  
  - Specificity: 96%  

- **K-Nearest Neighbors (KNN)**:  
  - Accuracy: 92%  
  - Sensitivity: 78%  
  - Specificity: 97%  

---

## **Conclusion**

This project demonstrates the effectiveness of using **vocal signal data** and **machine learning** to detect **Parkinson's Disease** at early stages. The models and methods used here have the potential to serve as **non-invasive diagnostic tools** in telemedicine.

---


## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/parkison-prediction.git

## **Usage**

Run the main script to start training the models:

```bash
python ParkinsonPredictionML.py

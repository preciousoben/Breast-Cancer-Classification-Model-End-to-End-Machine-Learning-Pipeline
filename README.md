# Breast Cancer Classification Model: End to End Machine Learning Pipeline
This project builds a **breast cancer classification model** using structured numerical data consisting of diagnostic measurements such as `mean_radius`, `mean_texture`, and `mean_area`. These measurements, along with others, are used to predict the **diagnosis**—either **benign** or **malignant**.  

A range of machine learning models were developed, and the **Random Forest model** was identified as the best performer based on metrics like **accuracy, precision, recall, and F1-score**. Additionally, a **KNIME Analytics workflow** ensures reproducibility. The dataset is **tabular, with continuous numerical features and a binary target**, making it ideal for supervised learning tasks.  

All results and insights are available in the Jupyter notebook and can be reproduced using the KNIME workflow. A **sample prediction on unseen data** is also demonstrated. The dataset is available **upon request**.  

---

# **Breast Cancer Classification Model**  

This project uses diagnostic measurements to classify breast cancer as either **benign or malignant**. Through various models such as **Random Forest, Logistic Regression, SVM, Decision Tree,** and **k-NN**, we aim to identify the most accurate classification method. The **Random Forest model** performed best and was saved for future predictions, tested on unseen data to ensure reliability.  

---

## **Project Structure & Highlights**  

- **Technologies Used:** Python, Jupyter Notebook, KNIME Analytics  
- **Dataset:** Includes continuous numerical measurements like `mean_radius`, `mean_texture`, and `mean_smoothness`  
- **Data Type:** Tabular data with **numerical features** and a **binary categorical target (diagnosis)**  
- **Skills Demonstrated:**  
  - Data cleaning, handling missing data, and detecting duplicates  
  - Model evaluation using metrics (accuracy, precision, recall, F1-score)  
  - Prediction on unseen data with trained Random Forest model  
  - Workflow automation and reproducibility using **KNIME Analytics**  
  - Dataset available **upon request**  

---

## **Results Overview**  

| Model                | Accuracy | Precision | Recall  | F1 Score | Confusion Matrix           |  
|----------------------|----------|-----------|---------|----------|----------------------------|  
| Logistic Regression  | 0.912    | 0.931     | 0.931   | 0.931    | [[37, 5], [5, 67]]         |  
| Decision Tree        | 0.886    | 0.954     | 0.861   | 0.905    | [[39, 3], [10, 62]]        |  
| Random Forest        | 0.921    | 0.957     | 0.917   | 0.936    | [[39, 3], [6, 66]]         |  
| SVM                  | 0.895    | 0.885     | 0.958   | 0.920    | [[33, 9], [3, 69]]         |  
| k-NN                 | 0.895    | 0.905     | 0.931   | 0.918    | [[35, 7], [5, 67]]         |  

---

## **How to Explore the Project**  

1. **Interactive Jupyter Notebook**:  
   Access the complete code and analysis [here](https://colab.research.google.com/drive/1-FC5iz_yemSDGbfeAemV-cO-5B4r314N?usp=sharing) on Google Colab to explore the models, results, and predictions.  
   
2. **KNIME Analytics Workflow**:  
   Use the KNIME workflow for reproducibility and automation.  

---

## **Project Workflow**  

1. **Data Cleaning & Preparation**  
   - Checked for **missing values** and **duplicates**  
   - Standardized data types to ensure smooth processing  

2. **Exploratory Data Analysis (EDA)**  
   - Examined summary statistics and the median of the variables  
   - Verified data distribution and relationships between features  

3. **Model Training & Evaluation**  
   - Models trained: **Logistic Regression, Decision Tree, Random Forest, SVM, and k-NN**  
   - Evaluated on **accuracy, precision, recall, and F1-score**  
   - Confusion matrix used to assess performance across classes  

4. **Testing the Best Model (Random Forest)**  
   - Saved the Random Forest model as `random_forest_model.pkl`  
   - Tested the model on unseen data to validate its reliability  

5. **KNIME Workflow for Reproducibility**  
   - Developed an automated workflow to handle data preprocessing and model training  

---

## **Repository Content**  

| File/Link                     | Description                                  |  
|-------------------------------|----------------------------------------------|  
| `breast cancer data analysis.ipynb`   | Jupyter notebook containing the code        |  
| `random_forest_model.pkl`     | Saved Random Forest model                   |  
|  `Breast cancer classification model.py`    | Python Script                            |
|[Jupyter Notebook](https://colab.research.google.com/drive/1-FC5iz_yemSDGbfeAemV-cO-5B4r314N?usp=sharing)|	Viewable notebook on Google Colab|
| `Breast_cancer_classification.knmf`            | Workflow file for reproducibility            |  

---

## **How to Run the Code Locally**  

1. **Clone the repository**:  
   
2. **Install dependencies**:  

3. **Run the Jupyter notebook**:  
  

---

## **Dataset Availability**  
The dataset used for this project is available **upon request**.  

---

Let me know if anything needs further refinement!

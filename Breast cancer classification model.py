#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
data_df=pd.read_csv('Breast_cancer_data.csv')


# In[3]:


data_df.info()


# In[4]:


duplicate_rows = data_df[data_df.duplicated()]

if duplicate_rows.empty:
    print("No duplicates found.")
else:
    print("Duplicate rows:")
    print(duplicate_rows)


# In[5]:


data_df.head()


# In[6]:


print(data_df.isnull().sum())


# In[7]:


duplicate_rows = data_df[data_df.duplicated()]

if duplicate_rows.empty:
    print("No duplicates found.")
else:
    print("Duplicate rows:")
    print(duplicate_rows)


# In[8]:


data_df.describe()


# In[9]:


data_df.median()


# In[13]:


from sklearn.model_selection import train_test_split

# Defining features and target variable
X = data_df.drop('diagnosis', axis=1)
y = data_df['diagnosis']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Displaying the shapes of the splits to confirm
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# In[15]:


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, conf_matrix


# In[16]:


log_reg = LogisticRegression(random_state=42, max_iter=10000)
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
svm = SVC(random_state=42)
knn = KNeighborsClassifier()


# In[17]:


models = {
    'Logistic Regression': log_reg,
    'Decision Tree': decision_tree,
    'Random Forest': random_forest,
    'SVM': svm,
    'k-NN': knn
}

results = {}

for model_name, model in models.items():
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': conf_matrix
    }


results


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, conf_matrix

# Initializing models
log_reg = LogisticRegression(random_state=42, max_iter=10000)
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
svm = SVC(random_state=42)
knn = KNeighborsClassifier()

# Evaluating each model
models = {
    'Logistic Regression': log_reg,
    'Decision Tree': decision_tree,
    'Random Forest': random_forest,
    'SVM': svm,
    'k-NN': knn
}

results = {}

for model_name, model in models.items():
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': conf_matrix
    }


results_df = pd.DataFrame(results).T


print(results_df)


# In[20]:


import joblib


# In[22]:


model_filename = 'random_forest_model.pkl'
joblib.dump(random_forest, model_filename)

# Loading the trained model
loaded_rf_classifier = joblib.load(model_filename)

y_pred_loaded = loaded_rf_classifier.predict(X_test)

# Evaluating the loaded model
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
precision_loaded = precision_score(y_test, y_pred_loaded)
recall_loaded = recall_score(y_test, y_pred_loaded)
f1_loaded = f1_score(y_test, y_pred_loaded)
conf_matrix_loaded = confusion_matrix(y_test, y_pred_loaded)

print(f"Accuracy: {accuracy_loaded}")
print(f"Precision: {precision_loaded}")
print(f"Recall: {recall_loaded}")
print(f"F1 Score: {f1_loaded}")
print(f"Confusion Matrix:\n{conf_matrix_loaded}")


# In[24]:


#testing

import pandas as pd
import joblib


new_data = pd.DataFrame({
    'mean_radius': [15.0],
    'mean_texture': [20.0],
    'mean_perimeter': [90.0],
    'mean_area': [800.0],
    'mean_smoothness': [0.1]
})

model_filename = 'random_forest_model.pkl'
loaded_rf_classifier = joblib.load(model_filename)

# prediction
predictions = loaded_rf_classifier.predict(new_data)

# Print predictions
print("Predictions:", predictions)


# In[ ]:





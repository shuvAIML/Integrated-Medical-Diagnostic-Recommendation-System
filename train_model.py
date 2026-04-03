import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

print("Loading dataset...")
df = pd.read_csv('data/dataset.csv')

print("Preprocessing text data into numerical binary format...")
# 1. Fill empty cells with empty strings
df = df.fillna('')

# 2. Identify symptom columns (everything except the 'Disease' column)
symptom_columns = [col for col in df.columns if col != 'Disease']

# 3. Extract a master list of all unique symptoms in the entire dataset
all_symptoms = set()
for col in symptom_columns:
    for val in df[col]:
        val_str = str(val).strip().lower()
        if val_str != '':
            # Kaggle data often has weird formatting like ' _itching'
            clean_val = val_str.replace('_', ' ').strip()
            all_symptoms.add(clean_val)

all_symptoms = sorted(list(all_symptoms))
print(f"Found {len(all_symptoms)} unique symptoms. Converting to numerical matrix...")

# 4. Create a binary (1/0) numerical matrix
encoded_data = []
for index, row in df.iterrows():
    disease = row['Disease'].strip()
    row_dict = {'Disease': disease}
    
    # Initialize all symptoms to 0
    for sym in all_symptoms:
        row_dict[sym] = 0
        
    # Set to 1 if the symptom exists for this patient
    for col in symptom_columns:
        val_str = str(row[col]).strip().lower()
        if val_str != '':
            clean_val = val_str.replace('_', ' ').strip()
            row_dict[clean_val] = 1
            
    encoded_data.append(row_dict)

binary_df = pd.DataFrame(encoded_data)

# 5. Define Features (X - symptoms) and Target (y - disease)
X = binary_df.drop('Disease', axis=1)
y = binary_df['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training ML Model (This may take a few seconds)...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Save the model and the exact feature columns
os.makedirs('models', exist_ok=True)
with open('models/medical_rf_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('models/columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("Training Complete! Models saved successfully.")
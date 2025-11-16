import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1: Load the dataset
df = pd.read_csv("Water Quality Testing.csv")  # replace with your filename
print("Original data size:", len(df))

# Step 2: Drop rows with missing values in important columns
important_columns = ['pH','Temperature','Turbidity','DO']
df_clean = df.dropna(subset=important_columns)
print("After cleaning:", len(df_clean))

# Step 3: Define water quality based on thresholds
def classify_water(row):
    if 6.5 <= row['pH'] <= 8.5 and row['Temperature'] <= 25 and row['Turbidity'] < 1 and row['DO'] >= 6:
        return 'Pure' 
    else:
        return 'Impure'

df_clean['quality'] = df_clean.apply(classify_water, axis=1) #New column added

# Step 4: Prepare features and labels
X = df_clean[['pH','Temperature' ,'Turbidity', 'DO']]
y = df_clean['quality']

# Step 5: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train.value_counts())

# Step 6: Train the Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

import joblib
joblib.dump(clf, "water_quality_model.pkl")

df_clean['Predicted_Quality'] = clf.predict(X) #New column added 

# Step 8: Save classified data
pure_data = df_clean[df_clean['Predicted_Quality'] == 'Pure']
Impure_data = df_clean[df_clean['Predicted_Quality'] == 'Unpure']
pure_data.to_csv("all_pure_water_data.csv", index=False)
Impure_data.to_csv("all_impure_water_data.csv", index=False)
print("Saved classified pure and impure data from the entire dataset.")

# Step 7: Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

pH = float(input("Enter pH: "))
Temperature = float(input("Enter Temperature (°C): "))
Turbidity = float(input("Enter Turbidity (NTU): "))
DO = float(input("Enter Dissolved Oxygen (mg/L): "))

sample = pd.DataFrame([[pH , Temperature , Turbidity , DO]] , columns = ['pH','Temperature','Turbidity','DO'])
prediction = clf.predict(sample)
print("Predicted Water Quality:",prediction[0]) 

import pandas as pd
from model import train_model

# Train model
model, columns, accuracy = train_model()

print("\n--- Health Insurance Cost Predictor ---")
print(f"Model R² Score: {accuracy:.2f}\n")

# User input
age = int(input("Enter age: "))
bmi = float(input("Enter BMI: "))
children = int(input("Number of children: "))

sex = input("Sex (male/female): ").lower()
smoker = input("Smoker (yes/no): ").lower()
region = input("Region (northeast/northwest/southeast/southwest): ").lower()

# Prepare input data (match one-hot encoded columns)
user_data = {
    'age': age,
    'bmi': bmi,
    'children': children,
    'sex_male': 1 if sex == 'male' else 0,
    'smoker_yes': 1 if smoker == 'yes' else 0,
    'region_northwest': 1 if region == 'northwest' else 0,
    'region_southeast': 1 if region == 'southeast' else 0,
    'region_southwest': 1 if region == 'southwest' else 0
}

user_df = pd.DataFrame([user_data], columns=columns)

# Predict insurance cost
prediction = model.predict(user_df)

print(f"\nEstimated Insurance Cost: ${prediction[0]:.2f}")

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from notification import send_push_notification


# Global variable for class_labels and root
class_labels = None
root = None

# Load the pre-trained model
file_path = 'animal.csv'
df = pd.read_csv(file_path)

# Remove leading/trailing whitespaces from column names
df.columns = df.columns.str.strip().str.replace(' ', '_')

# Feature Engineering and Scaling (similar to what you did in the previous steps)
df = pd.get_dummies(df, columns=['Animal'], drop_first=True)
df['age_temperature_interaction'] = df['Age'] * df['Temperature']
df['age_bin'] = pd.cut(df['Age'], bins=[0, 5, 10, 15, 20], labels=['0-5', '6-10', '11-15', '16-20'])
scaler = MinMaxScaler()
df[['Age', 'Temperature']] = scaler.fit_transform(df[['Age', 'Temperature']])
le = LabelEncoder()
df['age_bin'] = le.fit_transform(df['age_bin'])

# One-hot encode categorical columns in the original dataset
df_encoded = pd.get_dummies(df, columns=['Symptom_1', 'Symptom_2', 'Symptom_3'])

# Drop original categorical columns
df = df.drop(['Symptom_1', 'Symptom_2', 'Symptom_3'], axis=1)

# Split the dataset
x = df_encoded.drop('Disease', axis=1)
y = df_encoded['Disease']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model Training using the best model from GridSearchCV
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Tkinter GUI
def predict_disease():
    global class_labels

     # Clear previous predictions and recommendations
    predicted_disease_label.config(text="")
    class_labels_label.config(text="")
    recommendation_label.config(text="")

    # Get the input data from the GUI
    age = float(age_entry.get())
    temperature = float(temp_entry.get())
    animal = animal_var.get()
    symptom_1 = symptom1_menu.get()
    symptom_2 = symptom2_menu.get()
    symptom_3 = symptom3_menu.get()

    # Preprocess the input data
    input_data = {
        'Age': [age],
        'Temperature': [temperature],
        'Symptom_1_' + symptom_1: [1],
        'Symptom_2_' + symptom_2: [1],
        'Symptom_3_' + symptom_3: [1],
        'age_temperature_interaction': [age * temperature],
        'age_bin': [int(age >= 5)] 
    }

    # Add one-hot encoded animal features
    animal_columns = [col for col in df_encoded.columns if col.startswith('Animal_')]
    input_data.update({col: [0] for col in animal_columns})
    input_data['Animal_' + animal] = [1]

    input_df = pd.DataFrame(input_data, columns=X_train.columns)

    # Make prediction
    prediction = model.predict(input_df)

    # Get class labels from the model
    class_labels = model.classes_

    # Update the global variable
    class_labels = list(class_labels)

    # Display the predicted disease
    predicted_disease_label.config(text=f"Predicted Disease: {prediction[0]}")
    
    # Display class labels
    class_labels_label.config(text=f"Class Labels: {', '.join(class_labels)}")

    # Provide recommendations based on the predicted disease
    recommendations = {
        'foot_and_mouth': 'Isolate the animal and consult a veterinarian.',
        'pneumonia': 'Keep the animal warm and provide proper ventilation.',
        'anthrax': 'Isolate the animal and take it for Vaccination',
        'blackleg': 'Consult your veterinarian and the cattle should be vaccinated and treated prophylactically with administration of penicillin (10,000 IU/kg, IM) to prevent new cases for as long as 14 days. Cattle should also be moved from affected pastures..',
        'lumpy_virus': 'Isolate the animal and consult a veterinarian and Vaccination with attenuated virus.',
        
        # Add more recommendations for other diseases
    }

    recommended_action = recommendations.get(prediction[0], 'No specific recommendation available.')

    # Display recommendation
    recommendation_label.config(text=f"Recommendation: {recommended_action}")

   
    # Send email notification
    send_push_notification("Livestock Health Alert", recommended_action)

# GUI setup
root = tk.Tk()
root.title("Livestock Health Monitor")
# Set the size of the main window
root.geometry("600x400") 

# Labels and Entry widgets
tk.Label(root, text="Age:").grid(row=0, column=0, pady=5)
age_entry = tk.Entry(root)
age_entry.grid(row=0, column=1 ,pady=5)

tk.Label(root, text="Temperature:").grid(row=1, column=0, pady=5)
temp_entry = tk.Entry(root)
temp_entry.grid(row=1, column=1, pady=5)

tk.Label(root, text="Animal:").grid(row=2, column=0, pady=5)
animal_columns = [col for col in df_encoded.columns if col.startswith('Animal_')]
animals = [col.replace('Animal_', '') for col in animal_columns]
animal_var = tk.StringVar()
animal_menu = ttk.Combobox(root, textvariable=animal_var, values=animals)
animal_menu.grid(row=2, column=1,pady=5)

tk.Label(root, text="Symptom_1:").grid(row=3, column=0, pady=5)
symptom1_var = tk.StringVar()
symptom1_menu = ttk.Combobox(root, textvariable=symptom1_var, values=[col.replace('Symptom_1_', '') for col in df_encoded.filter(like='Symptom_1').columns])
symptom1_menu.grid(row=3, column=1, pady=5)

tk.Label(root, text="Symptom_2:").grid(row=4, column=0, pady=5)
symptom2_var = tk.StringVar()
symptom2_menu = ttk.Combobox(root, textvariable=symptom2_var, values=[col.replace('Symptom_2_', '') for col in df_encoded.filter(like='Symptom_2').columns])
symptom2_menu.grid(row=4, column=1, pady=5)

tk.Label(root, text="Symptom_3:").grid(row=5, column=0, pady=5)
symptom3_var = tk.StringVar()
symptom3_menu = ttk.Combobox(root, textvariable=symptom3_var, values=[col.replace('Symptom_3_', '') for col in df_encoded.filter(like='Symptom_3').columns])
symptom3_menu.grid(row=5, column=1, pady=5)

# Label to display predicted disease
predicted_disease_label = tk.Label(root, text="")
predicted_disease_label.grid(row=7, column=0, columnspan=2, pady=10)

# Label to display class labels
class_labels_label = tk.Label(root, text="")
class_labels_label.grid(row=8, column=0, columnspan=2, pady=10)

# Add a Label for displaying recommendations
recommendation_label = tk.Label(root, text="")
recommendation_label.grid(row=9, column=0, columnspan=2, pady=10)

# Button for prediction
predict_button = tk.Button(root, text="Predict Disease", command=predict_disease, bg="blue", fg="white")
predict_button.grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()

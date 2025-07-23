from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from werkzeug.security import generate_password_hash, check_password_hash
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Mock user data with hashed passwords
users = {'username': generate_password_hash('password')}

# Load the pre-trained model
file_path = 'animal.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.replace(' ', '_')
df = pd.get_dummies(df, columns=['Animal'], drop_first=True)
df['age_temperature_interaction'] = df['Age'] * df['Temperature']
df['age_bin'] = pd.cut(df['Age'], bins=[0, 5, 10, 15, 20], labels=['0-5', '6-10', '11-15', '16-20'])
scaler = MinMaxScaler()
df[['Age', 'Temperature']] = scaler.fit_transform(df[['Age', 'Temperature']])
le = LabelEncoder()
df['age_bin'] = le.fit_transform(df['age_bin'])
df_encoded = pd.get_dummies(df, columns=['Symptom_1', 'Symptom_2', 'Symptom_3'])
df = df.drop(['Symptom_1', 'Symptom_2', 'Symptom_3'], axis=1)
x = df_encoded.drop('Disease', axis=1)
y = df_encoded['Disease']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Function to predict disease 
def predict_disease(data):
    input_data = {
        'Age': [data['age']],
        'Temperature': [data['temperature']],
        'Symptom_1_' + data['symptom1']: [1],
        'Symptom_2_' + data['symptom2']: [1],
        'Symptom_3_' + data['symptom3']: [1],
        'age_temperature_interaction': [data['age'] * data['temperature']],
        'age_bin': [int(data['age'] >= 5)]  # adjust the threshold as needed
    }
    animal_columns = [col for col in df_encoded.columns if col.startswith('Animal_')]
    input_data.update({col: [0] for col in animal_columns})
    input_data['Animal_' + data['animal']] = [1]
    input_df = pd.DataFrame(input_data, columns=X_train.columns)
    prediction = model.predict(input_df)[0]
    recommendations = "No recommendations available."  # Add your recommendation logic here
    return prediction, recommendations

# Homepage
@app.route('/')
def home():
    return render_template('index.html')

# Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users.get(username) and check_password_hash(users[username], password):
            return redirect(url_for('home'))
        else:
            return render_template('login.html', message='Invalid credentials')
    return render_template('login.html', message=None)

# Registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return render_template('register.html', message='Username already exists.')
        users[username] = generate_password_hash(password)
        return redirect(url_for('login'))
    return render_template('register.html', message=None)

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = {
            'age': float(request.form.get('age')),
            'temperature': float(request.form.get('temperature')),
            'animal': request.form.get('animal'),
            'symptom1': request.form.get('symptom1'),
            'symptom2': request.form.get('symptom2'),
            'symptom3': request.form.get('symptom3')
        }
        # Check if age is not empty
        if data['age']:
            try:
                age = float(data['age'])
            except ValueError:
                return jsonify({'error': 'Invalid age value'}), 400
        else:
            return jsonify({'error': 'Age is required'}), 400

        prediction, recommendations = predict_disease(data)
        return jsonify({'prediction': prediction, 'recommendations': recommendations})
        

if __name__ == '__main__':
    app.run(debug=True)

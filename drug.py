import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import plotly.express as px


# Load the dataset
df = pd.read_csv('drug.csv')

# Data preprocessing
def preprocess_data(df):
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'M' else 0)
    df['BP'] = df['BP'].map({'LOW': 0, 'NORMAL': 1, 'HIGH': 2})
    df['Cholesterol'] = df['Cholesterol'].map({'NORMAL': 0, 'HIGH': 1})
    df['Drug'] = df['Drug'].map({'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'drugY': 4})
    return df

df = preprocess_data(df)

# Model building
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = df['Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train and save the model if not already saved
model_path = 'drug.pkl'
try:
    loaded_model = joblib.load(model_path)
    if not isinstance(loaded_model, DecisionTreeClassifier):
        raise ValueError("Loaded model is not a DecisionTreeClassifier")
except (FileNotFoundError, ValueError):
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    loaded_model = model

# Streamlit app
st.title('Drug Classification App')

# Sidebar
st.sidebar.header('Patient Information')
age = st.sidebar.slider('Age', 15, 75, 50)
sex = st.sidebar.selectbox('Sex', ['Female', 'Male'])
sex = 1 if sex == 'Male' else 0
bp = st.sidebar.selectbox('Blood Pressure', ['LOW', 'NORMAL', 'HIGH'])
bp = {'LOW': 0, 'NORMAL': 1, 'HIGH': 2}[bp]
cholesterol = st.sidebar.selectbox('Cholesterol', ['NORMAL', 'HIGH'])
cholesterol = 1 if cholesterol == 'HIGH' else 0
na_to_k = st.sidebar.slider('Na_to_K', 5.0, 40.0, 15.0)

# Prediction
new_patient = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'BP': [bp],
    'Cholesterol': [cholesterol],
    'Na_to_K': [na_to_k]
})

# Debug: Display the new patient data
st.write("New patient data for prediction:")
st.write(new_patient)

try:
    prediction = loaded_model.predict(new_patient)[0]
    predicted_drug = {0: 'drugA', 1: 'drugB', 2: 'drugC', 3: 'drugX', 4: 'drugY'}[prediction]

    # Display prediction
    st.subheader('Prediction')
    st.write(f'The predicted drug for the patient is: {predicted_drug}')
except Exception as e:
    st.write(f"An error occurred during prediction: {e}")

# Debug: Display model type
st.write(f"Loaded model type: {type(loaded_model)}")


required_columns = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
mmodel = joblib.load(model_path)


if all(column in df.columns for column in required_columns):
    # Predict the customer group
    X = df[required_columns]
    predictions = mmodel.predict(X)
    df['Drug_pred'] = predictions

    st.write("Predictions:")
    st.write(df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug', 'Drug_pred']])



# Graphs
st.subheader("Visualizations")
    
# Histogram of Age
fig_age = px.histogram(df, x='Age', nbins=20, title='Age Distribution')
st.plotly_chart(fig_age)
    
# Histogram of Na_to_K
fig_na_to_k = px.histogram(df, x='Na_to_K', nbins=20, title='Na_to_K Distribution')
st.plotly_chart(fig_na_to_k)

# Bar plot of Sex
sex_counts = df['Sex'].value_counts().reset_index()
sex_counts.columns = ['Sex', 'Count']
fig_sex = px.bar(sex_counts, x='Sex', y='Count', labels={'Sex': 'Sex', 'Count': 'Count'}, title='Sex Distribution')
st.plotly_chart(fig_sex)
    
# Bar plot of BP
bp_counts = df['BP'].value_counts().reset_index()
bp_counts.columns = ['BP', 'Count']
fig_bp = px.bar(bp_counts, x='BP', y='Count', labels={'BP': 'Blood Pressure', 'Count': 'Count'}, title='Blood Pressure Distribution')
st.plotly_chart(fig_bp)
    
# Bar plot of Cholesterol
cholesterol_counts = df['Cholesterol'].value_counts().reset_index()
cholesterol_counts.columns = ['Cholesterol', 'Count']
fig_cholesterol = px.bar(cholesterol_counts, x='Cholesterol', y='Count', labels={'Cholesterol': 'Cholesterol', 'Count': 'Count'}, title='Cholesterol Distribution')
st.plotly_chart(fig_cholesterol)
    
# Bar plot of Drug
drug_counts = df['Drug'].value_counts().reset_index()
drug_counts.columns = ['Drug', 'Count']
fig_drug = px.bar(drug_counts, x='Drug', y='Count', labels={'Drug': 'Drug', 'Count': 'Count'}, title='Drug Distribution')
st.plotly_chart(fig_drug)
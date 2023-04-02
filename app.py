import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from flask import Flask, request,render_template, url_for
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from flask import jsonify


model = tf.keras.models.load_model('model_1.h5')


def preprocess_input_data(df):
    import psycopg2

    # Replace the values in these variables with your own connection details
    host = "deeplearning.cz3vh7zutvpa.us-east-1.rds.amazonaws.com"
    port = "5432"
    dbname = "postgres"
    user = "postgres"
    password = "postgres"

    # Connect to the database
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )

    # Do something with the database
    # Execute a SQL query to retrieve the data from the customer_data table
    cur = conn.cursor()
    cur.execute("SELECT * FROM customer_data")

    # Fetch all the rows and convert the data into a Pandas DataFrame
    rows = cur.fetchall()
    df1 = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
    print(df1)

    # Close the database connection
    conn.close()

    df1.drop(columns=['Churn', 'customerID'], inplace=True)
    required_columns = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'tenure', 'MonthlyCharges', 'TotalCharges'
    ]

    df = df1.append(df, ignore_index=True)
    # Check if all required columns are present in the input data
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Input data is missing the following columns: {', '.join(missing_columns)}")

    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float, errors='raise')
    df['TotalCharges'] = df['TotalCharges'].astype(float, errors='raise')

    # Define the categorical and numerical columns
    categorical_columns = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]

    # Encode categorical features
    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])

    # Scale numerical features
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df.tail(1).values.tolist()[0]




app = Flask(__name__, static_url_path='',
            static_folder='./static',
            template_folder='./templates')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    print(request.json)
    # input_data_dict = {key: [value] for key, value in request.form.items()}
    # print(input_data_dict)
    # input_data_df = pd.DataFrame(input_data_dict)
    input_data_dict = request.json
    input_data_df = pd.DataFrame(input_data_dict, index=[0])
    print(input_data_df)
    preprocessed_data = preprocess_input_data(input_data_df)
    print(preprocessed_data)
    prediction = model.predict([preprocessed_data])
    print(prediction)
    threshold = 0.5
    if prediction[0][0] > threshold:
        churn_prediction = 'Yes'
    else:
        churn_prediction = 'No'
    print(churn_prediction)
    prediction_text = f'Churn Prediction: {churn_prediction}'
    # return render_template_string(template, prediction_text=prediction_text)
    return jsonify(churn_prediction=churn_prediction)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

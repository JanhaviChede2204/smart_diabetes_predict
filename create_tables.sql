-- Create Database
CREATE DATABASE smart_diabetes_predict;

USE smart_diabetes_predict;

-- Create table for storing predictions
CREATE TABLE DIABETES_PREDICTIONS (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_name VARCHAR(100),
    age INT,
    input_features JSON,
    predicted_label VARCHAR(20),
    predicted_prob FLOAT,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

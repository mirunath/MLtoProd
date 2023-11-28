# Wind Turbine Anomaly Detection

A cutting-edge modern factory is producing machine components for wind turbines. 
The factory is equipped with numerous sensors which track the production cycle. 
An anomaly detection system that integrates with the productive system is developed and implemented. 
Temperature, humidity, and sound volume are good indicators for an anomaly in the production cycle and a faulty produced item. 
The system must be able to process data streams, as the sensors in the factory measure data continuously. 
The implemented model should take these measurements over a standardized API and respond with a prediction score for an anomaly. 

# Approach

## 1. Data Ingestion

In Data Ingestion phase the data is first generated.
Then the data is split into training and testing.

## 2. Model Training

In this phase the KNN classifier is chosen and the model is created and saved as a pickle file.
The model is created directly in the Flask app.

## 3. Flask App Creation

The Flask app is created as an REST API to connect the turbine sensor and to automate monitoring.
It exposes an endpoint where it receives data and interprets it according to the model.

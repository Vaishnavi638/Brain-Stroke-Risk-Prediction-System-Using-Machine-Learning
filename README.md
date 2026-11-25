# Brain Stroke Risk Prediction System Using Machine Learning

## 1. Project Overview

The **Brain Stroke Risk Prediction System** is a machine learning-based web application developed to estimate an individual's risk of brain stroke using medical and lifestyle-related parameters.

The application uses a trained Machine Learning model built with Python and Scikit-learn to predict stroke probability based on patient information such as age, gender, hypertension, heart disease, glucose level, BMI, smoking status, and other health indicators.

The system integrates Machine Learning, healthcare analytics, web development, and database management into a unified platform. It provides patient record management, interactive health dashboards, personalized health recommendations, and automated PDF report generation.

The primary objective is to demonstrate how predictive analytics and Machine Learning can support healthcare risk assessment and preventive healthcare initiatives.

## 2. Key Features

- **User Authentication:** Login functionality for controlled access to the application.
- **Patient Information Management:** Collects and manages personal and medical information.
- **Stroke Risk Prediction:** Uses a trained Scikit-learn model to estimate the probability of stroke based on patient health parameters.
- **Health Analytics Dashboard:** Presents important health indicators, including BMI, blood glucose levels, and blood pressure.
- **Personalized Health Recommendations:** Provides health-related suggestions based on individual medical parameters.
- **Database Management:** Maintains patient records and prediction history using SQLite and Flask SQLAlchemy ORM.
- **PDF Report Generation:** Generates downloadable reports containing patient details, health analysis, prediction results, and recommendations.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Backend Framework | Flask |
| Database | SQLite |
| ORM | SQLAlchemy |
| Frontend | HTML, CSS, Bootstrap |
| Data Visualization | Matplotlib |
| PDF Generation | ReportLab |

## 4. Application Workflow

1. **User Authentication:** Users log in to access the application.
2. **Patient Data Collection:** Personal details and medical parameters are entered into the system.
3. **Machine Learning Prediction:** Patient data is passed to the trained ML model for stroke risk estimation.
4. **Health Analytics:** The application analyzes health indicators and displays relevant information through a dashboard.
5. **Recommendations:** Health-related suggestions are generated based on the patient's health parameters.
6. **Report Generation:** A PDF report is generated containing patient information, prediction results, and recommendations.
7. **Data Storage:** Patient records and prediction history are maintained in the SQLite database.

## 5. Machine Learning Component

The application integrates a supervised Machine Learning model developed using Scikit-learn.

The model uses patient health and lifestyle attributes to estimate the probability of stroke occurrence.

**Prediction Inputs Include:**
- Age and gender
- Hypertension and heart disease history
- Average glucose level
- Body Mass Index (BMI)
- Smoking status
- Residence type and other relevant parameters

**Prediction Output:** Estimated stroke probability based on the trained model.

## 6. Database Integration

SQLite is used for persistent data storage, with Flask SQLAlchemy ORM handling database interactions.

The database maintains:
- Patient personal information
- Medical information
- User authentication details
- Stroke prediction history


## 9. Project Objective

This project demonstrates the practical integration of Machine Learning and web technologies in healthcare analytics. It provides an application-level implementation of predictive modeling, patient data management, health indicator visualization, and automated reporting.

## 10. Future Enhancements

- Integrate additional Machine Learning algorithms for model comparison.
- Improve prediction performance through hyperparameter tuning.
- Introduce advanced healthcare data visualization.
- Extend the system with additional patient health indicators.
- Explore real-time healthcare data integration.



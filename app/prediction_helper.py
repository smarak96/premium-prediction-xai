# codebasics ML course: codebasics.io, all rights reserved

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np

model_young = joblib.load("artifacts\model_young.joblib")
model_rest = joblib.load("artifacts\model_rest.joblib")
scaler_young = joblib.load("artifacts\scaler_young.joblib")
scaler_rest = joblib.load("artifacts\scaler_rest.joblib")

explainer_rest = joblib.load("artifacts/explainer_rest.joblib")
explainer_young = joblib.load("artifacts/explainer_young.joblib")

# Precompute and cache global SHAP values
def initialize_global_shap_values():
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]
    background_data_young = pd.DataFrame(explainer_young.data, columns=expected_columns)
    background_data_rest = pd.DataFrame(explainer_rest.data, columns=expected_columns)
    
    np.random.seed(42)  # Set seed for reproducibility
    global_shap_values_young = explainer_young(background_data_young)
    global_shap_values_rest = explainer_rest(background_data_rest)
    
    return global_shap_values_young, global_shap_values_rest

# Initialize once at module load
global_shap_values_young, global_shap_values_rest = initialize_global_shap_values()

def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)
    max_score = 14
    min_score = 0
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)
    return normalized_risk_score

def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]
    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    df = pd.DataFrame(0, columns=expected_columns, index=[0])
    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':
            df['age'] = value
        elif key == 'Number of Dependants':
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value
        elif key == "Genetical Risk":
            df['genetical_risk'] = value

    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
    df = handle_scaling(input_dict['Age'], df)

    return df

def handle_scaling(age, df):
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = None
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df.drop('income_level', axis='columns', inplace=True)

    return df

def predict(input_dict):
    input_df = preprocess_input(input_dict)

    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction[0])

def get_shap_explanation(input_dict, input_df):
    if input_dict['Age'] <= 25:
        explainer = explainer_young
    else:
        explainer = explainer_rest

    shap_values = explainer(input_df)
    return shap_values

# def get_global_shap_plot(age_group):
#     if age_group <= 25:
#         shap_values = global_shap_values_young
#     else:
#         shap_values = global_shap_values_rest

#     plt.figure()
#     shap.plots.beeswarm(shap_values, max_display=20, show=False)
#     return plt.gcf()
# Singapore-Resale-Flat-Prices-Predicting

# Singapore-Resale-Flat-Prices-Predicting

Project Overview:
This project aims to develop a machine learning model and deploy it as a user-friendly web application to predict the resale prices of flats in Singapore. The model is based on historical transaction data and is designed to assist both potential buyers and sellers in estimating the market value of flats.

Skills Acquired:
Data Wrangling
Exploratory Data Analysis (EDA)
Model Building
Web Application Development

Domain:
Real Estate

Problem Statement:
The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.

Motivation:
The resale flat market in Singapore is highly competitive, and it can be challenging to accurately estimate the resale value of a flat. There are many factors that can affect resale prices, such as location, flat type, floor area, and lease duration. A predictive model can help to overcome these challenges by providing users with an estimated resale price based on these factors.

Project Scope:
The project involves the following tasks:

1. Data Collection and Preprocessing: Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) for the years 1990 to the present. Preprocess the data to clean and structure it for machine learning.
2. Feature Engineering: Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date. Create any additional features that may enhance prediction accuracy.
3. Model Selection and Training: Choose an appropriate machine learning model for regression (e.g., linear regression, decision trees, or random forests). Train the model on the historical data, using a portion of the dataset for training.
4. Model Evaluation: Evaluate the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R2 Score.
5. Streamlit Web Application: Develop a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.). Utilize the trained machine learning model to predict the resale price based on user inputs.
6. Testing and Validation: Thoroughly test the deployed application to ensure it functions correctly and provides accurate predictions.
   
Results:
The project benefits both potential buyers and sellers in the Singapore housing market. Buyers can use the application to estimate resale prices and make informed decisions, while sellers can get an idea of their flat's potential market value. Additionally, the project demonstrates the practical application of machine learning in real estate and web development.

Project Structure:
Singapore Resale Flat Prices Prediction/
├── images/
│   ├── 1.jpg
│   └── 4.png
├── data/
│   └── resale_flat_data.csv
├── models/
│   └── regression_model.pkl
├── app.py
├── README.md
└── requirements.txt
Usage

Clone the repository:
git clone https://github.com/bgomathidevi/Singapore_Resale_Flat_Prices_Prediction.git
cd Singapore_Resale_Flat_Prices_Prediction

Install the required packages:
pip install -r requirements.txt

Run the Streamlit application:
streamlit run app.py

Features:
Home Page: Provides an overview of the project, including the problem statement and project goals.
Flat Prediction Page: Allows users to input flat details and predict the resale price.
Analysis Page: Displays the performance of different machine learning models and final observations.

Model Performance:
The accuracy of various machine learning models was evaluated:

  1. Linear Regressor Accuracy: 68.23%
  2. Random Forest Regressor Accuracy: 95.66%
  3. Decision Tree Regressor Accuracy: 92.10%
Based on the accuracy scores, the Random Forest Regressor was chosen as the final model due to its highest accuracy of 95.66%.

Conclusion:
The Singapore Resale Flat Prices Prediction project has successfully developed a high-performing predictive model and a user-friendly web application that benefits both buyers and sellers in the competitive Singapore housing market. The high accuracy of the Random Forest Regressor model indicates a strong potential for practical application, providing a valuable tool for estimating resale prices and demonstrating the impactful role of machine learning in real estate.

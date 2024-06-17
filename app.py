
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Streamlit page custom design
def streamlit_config():
    st.set_page_config(layout="wide")
    st.write("""
    <div style='text-align:center'>
        <h1 style='color:#009999;'> Singapore Resale Price Prediction </h1>
    </div>
    """, unsafe_allow_html=True)

# custom style for submit button - color and width
def style_submit_button():
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #367F89;
            color: black;
            width: 70%;
            margin: 20px auto;
            display: block;
        }
        </style>
    """, unsafe_allow_html=True)

# custom style for prediction result text - color and position
def style_prediction():
    st.markdown("""
        <style>
        .center-text {
            text-align: center;
            color: #20CA0C;
        }
        </style>
    """, unsafe_allow_html=True)

# Load the model
with open("G:\\PROJECT\\Singapore-Resale-Flat-Prices-Predicting\\regression_model.pkl.gz", 'rb') as f:
    model = pickle.load(f)

# Feature names
features = ['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model',
             'year', 'month_of_year', 'lease_commence_year',
            'remaining_lease_years', 'remaining_lease_months']

# Categorical variable mappings
categorical_mappings = {
    'town': {'SENGKANG': 20, 'PUNGGOL': 17, 'WOODLANDS': 24, 'YISHUN': 25,
             'TAMPINES': 22, 'JURONG WEST': 13, 'BEDOK': 1, 'HOUGANG': 11,
             'CHOA CHU KANG': 8, 'ANG MO KIO': 0, 'BUKIT MERAH': 4, 'BUKIT PANJANG': 5,
             'BUKIT BATOK': 3, 'TOA PAYOH': 23, 'PASIR RIS': 16, 'KALLANG/WHAMPOA': 14,
             'QUEENSTOWN': 18, 'SEMBAWANG': 19, 'GEYLANG': 10, 'CLEMENTI': 9,
             'JURONG EAST': 12, 'BISHAN': 2, 'SERANGOON': 21, 'CENTRAL AREA': 7,
             'MARINE PARADE': 15, 'BUKIT TIMAH': 6},
    
    'flat_type': {'4 ROOM': 3, '5 ROOM': 4, '3 ROOM': 2,
                  'EXECUTIVE': 5, '2 ROOM': 1, 'MULTI-GENERATION': 6,
                  '1 ROOM': 0},
    
    'storey_range': {'04 TO 06': 1, '07 TO 09': 2, '10 TO 12': 3, '01 TO 03': 0,
                     '13 TO 15': 4, '16 TO 18': 5, '19 TO 21': 6, '22 TO 24': 7,
                     '25 TO 27': 8, '28 TO 30': 9, '31 TO 33': 10, '34 TO 36': 11,
                     '37 TO 39': 12, '40 TO 42': 13, '43 TO 45': 14, '46 TO 48': 15,
                     '49 TO 51': 16},
    
    'flat_model': {'Model A': 8, 'Improved': 5, 'New Generation': 12, 'Premium Apartment': 13,
                   'Simplified': 16, 'Apartment': 3, 'Maisonette': 7, 'Standard': 17,
                   'DBSS': 4, 'Model A2': 10, 'Model A-Maisonette': 9, 'Adjoined flat': 2,
                   'Type S1': 19, 'Type S2': 20, 'Premium Apartment Loft': 14, 'Terrace': 18,
                   'Multi Generation': 11, '2-room': 0, 'Improved-Maisonette': 6, '3Gen': 1,
                   'Premium Maisonette': 15},
}

# Function to display the home page
def About_page():
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader(":violet[Problem Statement:]")
        st.write("""
            
            -Develop a machine learning model and build  a user-friendly web application to predict resale flat prices in Singapore, 
                assisting both potential buyers and sellers in estimating the market value based on historical transaction data.
            """)
    with col2:
        
        st.write("* **:red[Purpose]** : Predict the selling Flat Price.")
        st.write("* **:red[Techniques Used]** : Data Wrangling and Preprocessing, Exploratory Data Analysis (EDA), Model Building and Evaluation, Web Application Development.")
        st.write("* **:red[Algorithm]** : Random Forest Regression.")

    st.image(Image.open(r"G:\\PROJECT\\Singapore-Resale-Flat-Prices-Predicting\\4.png"), width=1000)    
        

# Function to display the flat prediction page
def Prediction_page():

    input_data = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        for feature in features[:len(features)//2]:
            if feature in categorical_mappings:
                selected_option = st.selectbox(f"Select {feature.capitalize()}:", options=list(categorical_mappings[feature].keys()))
                input_data[feature] = categorical_mappings[feature][selected_option]
            else:
                input_data[feature] = st.number_input(f"{feature.capitalize()}:")

    with col2:
        for feature in features[len(features)//2:]:
            if feature in categorical_mappings:
                selected_option = st.selectbox(f"Select {feature.capitalize()}:", options=list(categorical_mappings[feature].keys()))
                input_data[feature] = categorical_mappings[feature][selected_option]
            else:
                input_data[feature] = st.number_input(f"{feature.capitalize()}:")

    with col3:
        if st.button("Predict"):
            input_array = np.array([input_data[feature] for feature in features]).reshape(1, -1)
            prediction = model.predict(input_array)

            # Display the prediction result
            prediction_scale = np.exp(prediction[0])
            st.subheader("Prediction Result:")
            st.write(f"The predicted house price is: {prediction_scale:,.2f} INR")

# Function to display the analysis page
def Conclusion_page():
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader(":violet[Model Performance]")
        st.write("""
        The accuracy of various machine learning models was evaluated:
        - Linear Regressor Accuracy: **:red[68.23%]**
        - Random Forest Regressor Accuracy: **:red[95.66%]**
        - Decision Tree Regressor Accuracy: **:red[92.10%]**

        Based on the accuracy scores, the **:red[Random Forest Regressor]** was chosen as the final model due to its highest accuracy of **:red[95.66%]**.
        """)

    with col2:
        st.subheader(":violet[Final Observations]")
        st.write("""
        - The Singapore Resale Flat Prices Prediction project has successfully developed a high-performing predictive model and a user-friendly web application that benefits both buyers and sellers in the competitive Singapore housing market.
        - The high accuracy of the Random Forest Regressor model indicates a strong potential for practical application, providing a valuable tool for estimating resale prices and demonstrating the impactful role of machine learning in real estate.
        """)

    st.image(Image.open(r"G:\\PROJECT\\Singapore-Resale-Flat-Prices-Predicting\\1.jpg"), width=700) 

# Configure the Streamlit app layout and style
streamlit_config()
style_submit_button()
style_prediction()

# Create the navigation menu in the sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Flat Prediction", "Analysis"],
        icons=["house", "graph-up", "bar-chart"],
        menu_icon="cast",
        default_index=0,
    )

# Display the selected page
if selected == "Home":
    About_page()
elif selected == "Flat Prediction":
    Prediction_page()
elif selected == "Analysis":
    Conclusion_page()


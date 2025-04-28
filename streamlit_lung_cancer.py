import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image

# Konfigurasi halaman dan judul
st.set_page_config(page_title="Portfolio", layout="wide", initial_sidebar_state="auto")
st.write("""
# PREDICTION DISEASE USING CLASSIFICATION
""")
st.write("""
Lung cancer is a kind of cancer that starts as a growth of cells in the lungs. The lungs are two spongy organs in the chest that control breathing.
Lung cancer is the leading cause of cancer deaths worldwide.
    """)

# Memanggil model yang sudah dilatih
with open("lung_cancer_logistic_model.pkl", "rb") as file:
    model_loaded = pickle.load(file)
st.image("lungcancer.png", caption="Lung Cancer")
# Menampilkan sidebar
st.sidebar.title("Choose what kind of disease you want to predict!")
use_case = st.sidebar.selectbox("Disease", ("Lung Cancer","Heart Disease(Coming soon)","Diabetes(Coming Soon)"))
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,col11,col12,col13,col14,col15 = st.columns(15, border=False, vertical_alignment='center')
def input_user_lungcancer_disease():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    if gender == "Male":
        gender = 0
        with col1:
            st.image("male.jpg", width=75, use_container_width=True)
    else:
        gender = 1
        with col1:
            st.image("female.jpg", width=75, use_container_width=True)
    AGE = st.sidebar.number_input("Age", min_value=0, max_value=100, value=50)
    if AGE <=18 :
        with col2:
            st.image('young.png', width=75, use_container_width=True)
    elif AGE >18 and AGE < 60 :
        with col2:
            st.image('adult.png', width=75, use_container_width=True)
    else:
        with col2:
            st.image('old.jpg',use_container_width=True,clamp=True,width=30)
    SMOKING = st.sidebar.selectbox("Smoking", ("Yes", "No"))
    if SMOKING == "Yes":
        SMOKING = 2
        with col3:
            st.image("smoking.jpg", width=75, use_container_width=True)
    else:
        SMOKING = 1
        with col3:
            st.image("no_smoking.png", width=75, use_container_width=True)
    Yellow_Fingers = st.sidebar.selectbox("Yellow Fingers", ("Yes", "No"))
    if Yellow_Fingers == "Yes":
        Yellow_Fingers = 2
        with col4:
            st.image("yellow.jpg", width=100, use_container_width=True)
    else:
        Yellow_Fingers = 1
        with col4:
            st.image("normal_fingers.jpg", width=100, use_container_width=True)
    Anxiety = st.sidebar.selectbox("Anxiety", ("Yes", "No"))
    if Anxiety == "Yes":
        Anxiety = 2
        with col5:
            st.image("anxiety.jpg",use_container_width=False)
    else:
        Anxiety= 1
        with col5:
            st.image("not_anxiety.jpg", use_container_width=False)
    Peer_pressure = st.sidebar.selectbox("Peer Pressure", ("Yes", "No"))
    if Peer_pressure == "Yes":
        Peer_pressure = 2
        with col6:
            st.image("peer_pressure.jpg",use_container_width=False)
    else:
        Peer_pressure= 1
        with col6:
            st.image("not_pressured.jpg", use_container_width=False)        
    CHRONIC_DISEASE = st.sidebar.selectbox("Chronic Disease", ("Yes", "No"))
    if CHRONIC_DISEASE == "Yes":
        CHRONIC_DISEASE = 2
        with col7:
            st.image("chronic.jpg",use_container_width=False)
    else:
        CHRONIC_DISEASE= 1
        with col7:
            st.image("not_chronic.jpg", use_container_width=False) 
    FATIGUE = st.sidebar.selectbox("Fatigue", ("Yes", "No"))
    if FATIGUE == "Yes":
        FATIGUE = 2
        with col8:
            st.image("fatigue.jpg",use_container_width=False)
    else:
        FATIGUE= 1
        with col8:
            st.image("not_fatigue.jpg", use_container_width=False) 
    ALLERGY = st.sidebar.selectbox("Allergy", ("Yes", "No"))
    if ALLERGY == "Yes":
        ALLERGY = 2
        with col9:
            st.image("allergy.jpg",use_container_width=False)
    else:
        ALLERGY= 1
        with col9:
            st.image("not_allergy.gif", use_container_width=False) 
    WHEEZING = st.sidebar.selectbox("Wheezing", ("Yes", "No"))
    if WHEEZING == "Yes":
        WHEEZING = 2
        with col10:
            st.image("wheezing.png",use_container_width=False)
    else:
        WHEEZING= 1
        with col10:
            st.image("not_wheezing.png", use_container_width=False) 
    ALCOHOL= st.sidebar.selectbox("Alcohol", ("Yes", "No"))
    if ALCOHOL == "Yes":
        ALCOHOL = 2
        with col11:
            st.image("alcohol.png",use_container_width=False)
    else:
        ALCOHOL= 1
        with col11:
            st.image("no_alcohol.png", use_container_width=False)
    COUGHING= st.sidebar.selectbox("Coughing", ("Yes", "No"))
    if COUGHING == "Yes":
        COUGHING = 2
        with col12:
            st.image("coughing.png",use_container_width=False)
    else:
        COUGHING= 1
        with col12:
            st.image("not_coughing.jpg", use_container_width=False)  
    SHORTNESS_OF_BREATH= st.sidebar.selectbox("Shortness of Breath", ("Yes", "No"))
    if SHORTNESS_OF_BREATH == "Yes":
        SHORTNESS_OF_BREATH = 2
        with col13:
            st.image("shortness_of_breath.png",use_container_width=False)
    else:
        SHORTNESS_OF_BREATH= 1
        with col13:
            st.image("not_shortness_of_breath.jpg", use_container_width=False)  
    SWALLOWING_DIFFICULTY= st.sidebar.selectbox("Swallowing Difficulty", ("Yes", "No"))
    if SWALLOWING_DIFFICULTY == "Yes":
        SWALLOWING_DIFFICULTY = 2
        with col14:
            st.image("swallow_difficult.png",use_container_width=False)
    else:
        SWALLOWING_DIFFICULTY= 1
        with col14:
            st.image("not_swallow_hard.png", use_container_width=False)
    CHEST_PAIN= st.sidebar.selectbox("Chest Pain", ("Yes", "No"))
    if CHEST_PAIN == "Yes":
        CHEST_PAIN = 2
        with col15:
            st.image("chest_pain.png",use_container_width=False)
    else:
        CHEST_PAIN= 1
        with col15:
            st.image("not_chest_pain.png", use_container_width=False)          
    data = { 'GENDER':gender,
            'AGE':AGE,
            'SMOKING':SMOKING,
            'YELLOW_FINGERS':Yellow_Fingers,
            'ANXIETY':Anxiety,
            'PEER_PRESSURE':Peer_pressure,
            'CHRONIC DISEASE':CHRONIC_DISEASE,
            'FATIGUE ':FATIGUE,
            'ALLERGY ':ALLERGY,
            'WHEEZING':WHEEZING,
            'ALCOHOL CONSUMING':ALCOHOL,
            'COUGHING':COUGHING,
            'SHORTNESS OF BREATH':SHORTNESS_OF_BREATH,
            'SWALLOWING DIFFICULTY':SWALLOWING_DIFFICULTY,
            'CHEST PAIN':CHEST_PAIN,
            }
    features = pd.DataFrame(data, index=[0])
    return features

# Function untuk menampilkan per use case
def lungcancer():
    st.sidebar.title("Input Data")
    st.sidebar.write("Please input your real condition, so we can predict:")
    input_data = input_user_lungcancer_disease()

    if st.sidebar.button("Check!"):
        df = input_data
        prediction = model_loaded.predict(df)
        st.write("Prediction:")
        result = [ "Unfortunalety you are potentially infected by Lung Cancer, please go to the doctor for futher treatment" if prediction == 1 else "Yeay, you are healthy enough. Keep your healthy lifestyle!"]
        output = str(result[0])
        with st.spinner("Loading..."):
            time.sleep(2)
            st.success(output)
        if prediction == 1:
            st.link_button(url='https://www.mayoclinic.org/diseases-conditions/lung-cancer/symptoms-causes/syc-20374620#:~:text=Lung%20cancer%20is%20a%20kind,greatest%20risk%20of%20lung%20cancer.',label="Clik here to get more information about lung cancer")
if use_case == "Lung Cancer":
    lungcancer()
elif use_case == "Diabetes":
    diabetes()
else :
    heart_disease()

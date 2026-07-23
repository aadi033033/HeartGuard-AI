import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# Load model
model = joblib.load("KNN_Heart.pkl")
scaler = joblib.load("scaler_Heart.pkl")
expected_columns = joblib.load("Columns_Heart.pkl")


# Page configuration
st.set_page_config(
    page_title="Heart Disease AI Predictor",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<div class="hero">
<h1>❤️ HeartGuard AI</h1>
<p>AI Powered Heart Disease Risk Prediction System</p>
</div>
""",
unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>

body {
    background-color:#f4f7fb;
}


/* Main title */
.hero {
    background: linear-gradient(135deg,#ff416c,#ff4b2b);
    padding:35px;
    border-radius:25px;
    text-align:center;
    color:white;
    margin-bottom:30px;
}


.hero h1 {
    font-size:50px;
    margin-bottom:5px;
}


.hero p {
    font-size:22px;
}



/* Cards */
.card {
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 8px 25px rgba(0,0,0,0.08);
    margin:15px;
}



/* Section titles */
h2 {
color:#ff416c;
}



/* Button */
.stButton button {
    background:linear-gradient(
        90deg,#ff416c,#ff4b2b
    );

    color:white;
    font-size:22px;
    font-weight:bold;
    border-radius:15px;
    height:55px;
    width:100%;
    border:none;
}



/* Metric */
[data-testid="stMetricValue"]{
    font-size:35px;
    color:#ff416c;
}

</style>
""",
unsafe_allow_html=True)


# Header

st.markdown(
    "<div class='title'>❤️ Heart Disease AI Predictor</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Machine Learning based heart disease risk prediction</div>",
    unsafe_allow_html=True
)

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/822/822143.png",
        width=150
    )


    st.title(
        "❤️ HeartGuard AI"
    )


    st.write(
    """
    Built using:

    🤖 Logistic Regression

    🐍 Python

    📊 Scikit-learn

    🎨 Streamlit

    """
    )


    st.info(
    "This tool is for educational purposes only."
    )


st.write("")


# Input section

st.markdown(
    "<div class='card'>",
    unsafe_allow_html=True
)

st.subheader("👤 Patient Information")


col1, col2, col3 = st.columns(3)


with col1:

    Age = st.slider(
        "Age",
        1,
        100,
        50
    )

    Sex = st.selectbox(
        "Gender",
        ["M","F"]
    )


    ChestPainType = st.selectbox(
        "Chest Pain Type",
        ["ATA","NAP","ASY","TA"]
    )


with col2:

    RestingBP = st.number_input(
        "Resting Blood Pressure",
        80,
        200,
        120
    )

    Cholesterol = st.number_input(
        "Cholesterol",
        0,
        600,
        200
    )


    FastingBS = st.selectbox(
        "Fasting Blood Sugar",
        [0,1]
    )


with col3:

    RestingECG = st.selectbox(
        "Resting ECG",
        ["Normal","ST","LVH"]
    )


    MaxHR = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )


    ExerciseAngina = st.selectbox(
        "Exercise Angina",
        ["Y","N"]
    )


Oldpeak = st.slider(
    "Oldpeak",
    0.0,
    10.0,
    1.0
)


ST_Slope = st.selectbox(
    "ST Slope",
    ["Up","Flat","Down"]
)


st.markdown("</div>", unsafe_allow_html=True)



# Prediction button

st.write("")

if st.button("🔍 Predict Heart Disease Risk"):


    input_data = pd.DataFrame({

        "Age":[Age],
        "Sex":[Sex],
        "ChestPainType":[ChestPainType],
        "RestingBP":[RestingBP],
        "Cholesterol":[Cholesterol],
        "FastingBS":[FastingBS],
        "RestingECG":[RestingECG],
        "MaxHR":[MaxHR],
        "ExerciseAngina":[ExerciseAngina],
        "Oldpeak":[Oldpeak],
        "ST_Slope":[ST_Slope]

    })


    # Encoding
    input_data = pd.get_dummies(input_data)


    # Match training columns
    input_data = input_data.reindex(
        columns=scaler.feature_names_in_,
        fill_value=0
    )


    input_scaled = scaler.transform(input_data)


    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)[0][1]


    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability*100,
            title={
                'text':
                "Heart Disease Risk %"
            },

            gauge={
                'axis':
                {
                    'range':[0,100]
                }

            }

        )
    )


    fig.update_layout(
        height=350
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.write("")


    # Result Card

    st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True
    )


    st.subheader("📊 Prediction Result")


    col1,col2 = st.columns(2)


    with col1:

        st.metric(
            "Risk Probability",
            f"{probability*100:.2f}%"
        )


    with col2:

        if probability < 0.3:
            st.success(
                """
                🟢 Low Risk

                Maintain:
                    - Regular exercise
                    - Balanced diet
                    - Annual health checkup
                    """
            )

        elif probability <0.7:
            st.warning(
                """
                🟡 Moderate Risk

                Consider:
                    - Reduce cholesterol intake
                    - Increase physical activity
                    - Consult doctor
                    """
            )
        else:
            st.error(
                """
                🔴 High Risk

                Recommended:
                    - Medical consultation
                    - Heart monitoring
                    - Lifestyle changes
                    """
            )


    st.progress(float(probability))


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )
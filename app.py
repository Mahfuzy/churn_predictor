import streamlit as st
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Try to get API key from Streamlit secrets first, then from environment variables
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is missing. Please set it in either .streamlit/secrets.toml or .env file.")
    st.stop()

client = Groq(api_key=groq_api_key)

# Initialize session state for dark mode and results
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "churn_prob" not in st.session_state:
    st.session_state.churn_prob = None
if "original_data" not in st.session_state:
    st.session_state.original_data = None
if "retention_strategy" not in st.session_state:
    st.session_state.retention_strategy = None

# Icon switch for dark/light mode
icon = "🔆" if st.session_state.dark_mode else "🌙"

# Custom CSS for button visibility and styling
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 16px;
        cursor: pointer;
        margin: 10px 0;
        display: block;
        z-index: 1000;
    }
    .stButton > button:hover {
        background-color: #45a049;
        border-color: #45a049;
    }
    .dark-mode-toggle {
        position: fixed;
        top: 10px;
        right: 10px;
        font-size: 24px;
        background: none;
        border: none;
        cursor: pointer;
        z-index: 1000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Apply dark/light mode styles
if st.session_state.dark_mode:
    st.markdown(
        """
        <style>
        body, .stApp { background-color: #0A192F; color: #E0E5EC; font-family: 'Segoe UI', sans-serif; }
        h1, h2, h3, p, label { color: white !important; }
        .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > select {
            background-color: #112240; 
            color: #E0E5EC; 
            border-radius: 8px; 
            padding: 10px; 
            border: 2px solid transparent;
            box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.2);
        }
        .stButton > button {
            background-color:rgb(46, 63, 80);
            border: 2px solid rgb(46, 63, 80);
        }
        .stButton > button:hover {
            background-color: rgb(41, 51, 61);
            border-color: rgb(41, 51, 61);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        body, .stApp { background-color: #ffffff; color: #333; }
        h1, h2, h3, p, label { color: #333 !important; }
        .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > select {
            background-color: #f8f9fa; 
            color: #333; 
            border-radius: 8px; 
            padding: 10px; 
            border: 1px solid #ccc;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        .stButton > button {
            background-color:rgb(113, 169, 224);
            border: 2px solid rgb(113, 169, 224);
        }
        .stButton > button:hover {
            background-color: rgb(65, 146, 228);
            border-color: rgb(65, 146, 228);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Hide Streamlit UI elements
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load pretrained model and scaler
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please train and save the model first.")
    st.stop()

def get_retention_strategy(churn_prob, original_data):
    """Generate retention strategy using Groq API."""
    prompt = f"""
    You are a banking customer retention expert. A customer has a {churn_prob:.2%} chance of churning. Their details are:
    - Credit Score: {original_data['credit_score']}
    - Country: {original_data['country']}
    - Gender: {original_data['gender']}
    - Age: {original_data['age']}
    - Tenure: {original_data['tenure']} years
    - Balance: ${original_data['balance']:,.2f}
    - Number of Products: {original_data['products_number']}
    - Has Credit Card: {'Yes' if original_data['credit_card'] else 'No'}
    - Active Member: {'Yes' if original_data['active_member'] else 'No'}
    - Estimated Salary: ${original_data['estimated_salary']:,.2f}

    Suggest specific retention strategies to reduce the likelihood of churn, focusing on actionable steps based on the customer's profile.
    """
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating retention strategy: {str(e)}"

def generate_draft_email(churn_prob, original_data):
    """Generate a personalized draft email based on churn probability."""
    prompt = f"""
    You are a banking customer service representative crafting a concise, engaging email to retain a customer at risk of leaving. The customer has a {churn_prob:.2%} chance of churning. Their details are:
    - Name: {original_data.get('name', 'Valued Customer')}
    - Credit Score: {original_data['credit_score']}
    - Country: {original_data['country']}
    - Age: {original_data['age']}
    - Tenure: {original_data['tenure']} years
    - Balance: ${original_data['balance']:,.2f}
    - Number of Products: {original_data['products_number']}
    - Has Credit Card: {'Yes' if original_data['credit_card'] else 'No'}
    - Active Member: {'Yes' if original_data['active_member'] else 'No'}
    - Estimated Salary: ${original_data['estimated_salary']:,.2f}

    Write a professional, empathetic email (200-250 words) that:
    1. Opens with gratitude for their {original_data['tenure']} years with the bank, addressing them by name.
    2. Offers 2-3 tailored incentives based on their profile (e.g., fee waivers for low balances, loyalty rewards for long tenure, or product upgrades for active members).
    3. Includes a specific call-to-action, such as scheduling a call with a retention specialist via a provided link or phone number.
    4. Uses a warm, professional tone and avoids generic phrases.
    5. Closes with a personalized sign-off from a fictional retention specialist (e.g., Light Yagami, Death Note Team).
    Ensure the email feels personal, actionable, and aligned with the customer's profile.
    """
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating draft email: {str(e)}"

# Streamlit UI
st.title("Bank Customer Churn Prediction")

with st.container():
    # Toggle dark/light mode
    if st.button(icon, key="dark_mode_toggle", help="Toggle Dark/Light Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.header("Customer Information")
    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Customer Name", value="Valued Customer")
        credit_score = st.number_input("Credit Score (300-850)", min_value=300, max_value=850, value=600)
        country = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=20, value=5)

    with col2:
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=5000.0)
        products_number = st.number_input("Number of Products", min_value=1, max_value=10, value=1)
        credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        active_member = st.selectbox("Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0)

    # Store original data for display
    original_data = {
        'name': name,
        'credit_score': credit_score,
        'country': country,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': credit_card == "Yes",
        'active_member': active_member == "Yes",
        'estimated_salary': estimated_salary
    }

    # Convert binary inputs to 0/1
    credit_card = 1 if credit_card == "Yes" else 0
    active_member = 1 if active_member == "Yes" else 0

    if st.button("Predict Churn"):
        # Create input DataFrame
        input_data = pd.DataFrame({
            'credit_score': [credit_score],
            'country': [country],
            'gender': [gender],
            'age': [age],
            'tenure': [tenure],
            'balance': [balance],
            'products_number': [products_number],
            'credit_card': [credit_card],
            'active_member': [active_member],
            'estimated_salary': [estimated_salary]
        })

        # One-hot encode categorical variables
        input_data = pd.get_dummies(input_data, columns=['country', 'gender'], drop_first=True)

        # Ensure all expected columns are present
        expected_columns = [
            'credit_score', 'age', 'tenure', 'balance', 'products_number',
            'credit_card', 'active_member', 'estimated_salary',
            'country_Germany', 'country_Spain', 'gender_Male'
        ]
        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[expected_columns]

        # Scale numerical features
        numerical_cols = ['credit_score', 'age', 'tenure', 'balance', 'estimated_salary']
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

        # Predict churn
        churn_prob = model.predict_proba(input_data)[:, 1][0]
        churn_pred = model.predict(input_data)[0]

        # Store results in session state
        st.session_state.churn_prob = churn_prob
        st.session_state.original_data = original_data
        st.session_state.retention_strategy = get_retention_strategy(churn_prob, original_data)

    # Display retained results if available
    if st.session_state.churn_prob is not None:
        st.header("Prediction Results")
        st.markdown(f"**Churn Probability**: {st.session_state.churn_prob:.2%}")
        if st.session_state.churn_prob > 0.5:
            st.error("High risk of churn.")
        else:
            st.success("Low risk of churn.")
        if st.session_state.retention_strategy:
            st.markdown("### Retention Strategy")
            st.markdown(f"<div class='stMarkdown'>{st.session_state.retention_strategy}</div>", unsafe_allow_html=True)

    # Separate button for generating draft email
    if st.session_state.churn_prob is not None and st.button("Generate Draft Email"):
        draft_email = generate_draft_email(st.session_state.churn_prob, st.session_state.original_data)
        st.markdown("### Draft Email")
        st.markdown(f"<div class='stMarkdown'>{draft_email}</div>", unsafe_allow_html=True)
        st.download_button(
            label="Download Email Draft",
            data=draft_email,
            file_name="customer_retention_email.txt",
            mime="text/plain"
        )

# Run on the correct port for deployment
if __name__ == "__main__":
    os.system(f"streamlit run app.py --server.port {os.getenv('PORT', '8080')} --server.address 0.0.0.0")
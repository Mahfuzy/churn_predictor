# Bank Customer Churn Prediction

A machine learning application that predicts customer churn probability and provides AI-powered retention strategies for banking institutions.

## Features

- Real-time churn prediction using XGBoost model
- AI-powered retention strategy recommendations using Groq API
- Interactive web interface with dark/light mode
- Comprehensive customer data analysis
- SMOTE-based handling of class imbalance
- Feature scaling and preprocessing

## Project Structure

```
├── app.py              # Main Streamlit application
├── train_model.py      # Model training script
├── model.joblib        # Trained XGBoost model
├── scaler.joblib       # Feature scaler
├── bank_churn.csv      # Dataset
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Dataset

The application uses a bank customer dataset with the following features:
- Customer ID
- Credit Score
- Country
- Gender
- Age
- Tenure
- Balance
- Products Number
- Credit Card Status
- Active Member Status
- Estimated Salary
- Churn Status

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bank-churn-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root and add your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

## Usage

1. Train the model:
```bash
python train_model.py
```

2. Run the application:
```bash
streamlit run app.py
```

3. Access the web interface at `http://localhost:8080`

## Model Training

The model training process includes:
- Data preprocessing and feature engineering
- SMOTE for handling class imbalance
- XGBoost classifier training
- Model evaluation using ROC-AUC and classification metrics

## Application Features

1. **Customer Information Input**
   - Credit Score (300-850)
   - Country selection
   - Gender selection
   - Age input
   - Tenure input
   - Balance input
   - Products number
   - Credit card status
   - Active member status
   - Estimated salary

2. **Churn Prediction**
   - Probability-based prediction
   - Risk level indication
   - AI-generated retention strategies

3. **User Interface**
   - Dark/light mode toggle
   - Responsive design
   - Input validation
   - Clear results display

## Technical Details

- **Frontend**: Streamlit
- **Machine Learning**: XGBoost, scikit-learn
- **Data Processing**: pandas, numpy
- **AI Integration**: Groq API
- **Model Persistence**: joblib

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


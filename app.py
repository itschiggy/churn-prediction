import streamlit as st
import pickle

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def load_model():
    """Loads the churn prediction model and vectorizer."""
    with open('churn-model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def predict_single(customer, dv, model):
    """Predicts churn probability for a single customer."""
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


def get_customer_info():
    """Collects customer information from user input."""
    customer = {}

    col1, col2 = st.columns(2)

    partner = col1.selectbox(" Does the customer have a partner?", ['yes', 'no'])
    customer['partner'] = partner

    monthly_charges = col1.slider('What are the monthly charges for the customer?', 15, 120, 10)
    customer['monthlycharges'] = monthly_charges

    tenure = col2.slider('How many months has the customer been with us?', 0, 75, 5)
    customer['tenure'] = tenure

    contract = col2.selectbox('Contract type', ['month-to-month', 'one_year', 'two_year'])
    customer['contract'] = contract

    return customer


# Main app logic
def main():
    st.set_page_config(page_title="Churn Prediction App", layout="wide")

    # Display app logo or title with improved styling
    st.title("Churn Prediction for Telecom Customers")
    # Consider adding an app logo using st.image() for visual appeal

    # Load pre-trained model and vectorizer
    dv, model = load_model()

    customer = get_customer_info()

    if st.button("Predict Churn Probability"):
        churn_prob = predict_single(customer, dv, model)

        # Display informative output with clear interpretation
        st.header(f"Churn Probability: {churn_prob:.2f}")

        if churn_prob >= 0.5:
            st.warning("High Churn Risk! The customer has a high probability of churning.")
            st.markdown(
                """We recommend considering targeted retention offers to incentivize the customer to stay with your company."""
            )
        else:
            st.success("Low Churn Risk! The customer is likely to remain a loyal customer.")

        # Optional: Show feature importances for interpretability

if __name__ == "__main__":
    main()

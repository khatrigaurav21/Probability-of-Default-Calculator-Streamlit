import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ... (Your existing functions: calculate_ebitda, calculate_ratios, assign_risk_scores, calculate_overall_score, determine_pd, generate_report)

st.title("Probability of Default (PD) Calculator with ML")

# ... (Your existing manual input and CSV upload sections)

# Training Data Upload
st.header("Upload Training Data (CSV)")
training_file = st.file_uploader("Training Data", type="csv")

model = None  # Initialize model variable
if training_file is not None:
    try:
        training_df = pd.read_csv(training_file)
        if 'Default' not in training_df.columns:
            st.error("Error: 'Default' column missing from training data.")
        else:
            X = training_df.drop('Default', axis=1)
            y = training_df['Default']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            if st.button("Train Model"):
                model = LogisticRegression(max_iter=1000) #increase max_iter
                model.fit(X_train, y_train)
                accuracy = accuracy_score(y_test, model.predict(X_test))
                st.write(f"Model Accuracy: {accuracy}")
                st.success("Model Trained and Saved!")
    except Exception as e:
        st.error(f"Error processing training data: {e}")

# Calculate PD Button
if st.button("Calculate PD"):
    ebitda = calculate_ebitda(revenue, cogs, opex)
    ratios = calculate_ratios(ebitda, total_debt, interest_expense, cash, short_term_debt)
    scores = assign_risk_scores(ratios)
    overall_score = calculate_overall_score(scores)
    pd_result = determine_pd(overall_score)

    inputs = {
        "Revenue": revenue,
        "COGS": cogs,
        "OPEX": opex,
        "Total Debt": total_debt,
        "Interest Expense": interest_expense,
        "Cash": cash,
        "Short Term Debt": short_term_debt,
    }

    ml_pd = None
    if model is not None:
        input_data = pd.DataFrame([inputs])
        ml_pd = model.predict_proba(input_data)[0][1] #get probability of default

    report = generate_report(inputs, ebitda, ratios, scores, overall_score, pd_result, ml_pd)
    st.markdown(report)

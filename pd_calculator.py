import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def calculate_ebitda(revenue, cogs, opex):
    """Calculates EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization)."""
    return revenue - cogs - opex

def calculate_ratios(ebitda, total_debt, interest_expense, cash, short_term_debt):
    """Calculates key financial ratios."""
    ratios = {
        "EBITDA/Total Debt": ebitda / total_debt if total_debt != 0 else 0,
        "Debt/EBITDA": total_debt / ebitda if ebitda != 0 else 0,
        "EBITDA/Interest Expense": ebitda / interest_expense if interest_expense != 0 else 0,
        "Cash/Short Term Debt": cash / short_term_debt if short_term_debt != 0 else 0,
    }
    return ratios

def assign_risk_scores(ratios):
    """Assigns risk scores based on financial ratios."""
    scores = {}
    scores["EBITDA/Total Debt Score"] = 3 if ratios["EBITDA/Total Debt"] > 0.5 else 2 if ratios["EBITDA/Total Debt"] > 0.2 else 1
    scores["Debt/EBITDA Score"] = 1 if ratios["Debt/EBITDA"] > 5 else 2 if ratios["Debt/EBITDA"] > 2 else 3
    scores["EBITDA/Interest Expense Score"] = 3 if ratios["EBITDA/Interest Expense"] > 2 else 2 if ratios["EBITDA/Interest Expense"] > 1 else 1
    scores["Cash/Short Term Debt Score"] = 3 if ratios["Cash/Short Term Debt"] > 1 else 2 if ratios["Cash/Short Term Debt"] > 0.5 else 1
    return scores

def calculate_overall_score(scores):
    """Calculates the overall risk score."""
    return sum(scores.values()) / len(scores)

def determine_pd(overall_score):
    """Determines the probability of default based on the overall risk score."""
    if overall_score > 2.5:
        return "Low Risk (PD < 2%)"
    elif overall_score > 1.8:
        return "Medium Risk (2% <= PD < 10%)"
    else:
        return "High Risk (PD >= 10%)"

def generate_report(inputs, ebitda, ratios, scores, overall_score, pd_result, ml_pd=None):
    """Generates a financial report."""
    report = f"""
    ## Financial Report

    **Financial Inputs:**
    {pd.DataFrame([inputs]).T.to_markdown()}

    **EBITDA:** {ebitda:.2f}

    **Key Ratios:**
    {pd.DataFrame([ratios]).T.to_markdown()}

    **Ratio Scores:**
    {pd.DataFrame([scores]).T.to_markdown()}

    **Overall Risk Score:** {overall_score:.2f}

    **Probability of Default (PD) (Ratio-based):** {pd_result}
    """
    if ml_pd is not None:
        report += f"\n\n**Probability of Default (PD) (ML-based):** {ml_pd:.4f}"

    report += """

    **Recommendations:**
    - **Debt Management:** Analyze Debt/EBITDA and EBITDA/Total Debt ratios. Consider refinancing or reducing debt if needed.
    - **Profitability:** Review EBITDA. Improve revenue and reduce costs if EBITDA is low.
    - **Liquidity:** Check Cash/Short Term Debt. Increase liquid assets or improve cash flow if needed.
    - **Interest Coverage:** Improve EBITDA/Interest Expense. Consider refinancing or improving earnings if needed.
    """
    return report

st.title("Probability of Default (PD) Calculator with ML")

# Manual Input Section
st.header("Manual Input")

revenue_help = "Total revenue generated by the business."
cogs_help = "Direct costs associated with producing goods or services."
opex_help = "Operating expenses, including selling, general, and administrative costs."
total_debt_help = "Total outstanding debt of the business."
interest_expense_help = "Interest paid on the debt."
cash_help = "Cash and cash equivalents held by the business."
short_term_debt_help = "Debt due within one year."

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("Revenue")
    st.info(revenue_help)
with col2:
    revenue = st.number_input("", value=1000000.0)

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("COGS")
    st.info(cogs_help)
with col2:
    cogs = st.number_input("", value=500000.0)

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("OPEX")
    st.info(opex_help)
with col2:
    opex = st.number_input("", value=300000.0)

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("Total Debt")
    st.info(total_debt_help)
with col2:
    total_debt = st.number_input("", value=500000.0)

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("Interest Expense")
    st.info(interest_expense_help)
with col2:
    interest_expense = st.number_input("", value=50000.0)

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("Cash")
    st.info(cash_help)
with col2:
    cash = st.number_input("", value=200000.0)

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("Short Term Debt")
    st.info(short_term_debt_help)
with col2:
    short_term_debt = st.number_input("", value=100000.0)

# CSV Upload Section
st.header("Upload Financial Data (CSV)")
uploaded_file = st.file_uploader("", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ["Revenue", "COGS", "OPEX", "Total Debt", "Interest Expense", "Cash", "Short Term Debt"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Error: Missing columns in CSV: {', '.join(missing_columns)}")
        else:
            revenue = df['Revenue'].iloc[0]
            cogs = df['COGS'].iloc[0]
            opex = df['OPEX'].iloc[0]
            total_debt = df['Total Debt'].iloc[0]
            interest_expense = df['Interest Expense'].iloc[0]
            cash = df['Cash'].iloc[0]
            short_term_debt = df['Short Term Debt'].iloc[0]
    except Exception as e:
        st.error(f"Error processing CSV: {e}")

# Training Data Upload
st.header("Upload Training Data (CSV)")
training_file = st.file_uploader("Training Data", type="csv")

if training_file is not None:
    try:
        training_df = pd.read_csv(training_file)
        if 'Default' not in training_df.columns:
            st.error("Error: 'Default' column missing from training data.")
        else:
            X = training_df.drop('Default', axis=1)
            y = training_df['Default']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import os
import sys
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import anthropic
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag.rag_engine import get_context

# =====================================================
# SETUP
# =====================================================

load_dotenv()
app = FastAPI(title="Finance Advisor AI", version="1.0.0")

# Load ML models
gmm         = joblib.load("models/gmm_model.pkl")
gmm_scaler  = joblib.load("models/gmm_scaler.pkl")
label_map   = joblib.load("models/gmm_label_map.pkl")
rf          = joblib.load("models/rf_model.pkl")
rf_scaler   = joblib.load("models/rf_scaler.pkl")
le          = joblib.load("models/rf_label_encoder.pkl")

# Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# =====================================================
# REQUEST SCHEMA
# =====================================================

class UserFinanceInput(BaseModel):
    monthly_income: float
    monthly_expense_total: float
    savings_rate: float
    debt_to_income_ratio: float
    credit_score: float
    loan_payment: float
    investment_amount: float
    discretionary_spending: float
    essential_spending: float
    emergency_fund: float
    question: str = "What should I do to improve my financial situation?"

# =====================================================
# ML PREDICTION
# =====================================================

def predict(user_input: UserFinanceInput) -> dict:
    # GMM — segment
    gmm_features = ["monthly_income", "savings_rate", "debt_to_income_ratio"]
    gmm_input = np.array([[
        user_input.monthly_income,
        user_input.savings_rate,
        user_input.debt_to_income_ratio
    ]])
    gmm_scaled = gmm_scaler.transform(gmm_input)
    cluster = gmm.predict(gmm_scaled)[0]
    segment = label_map[cluster]

    # RF — cash flow
    rf_input = np.array([[
        user_input.monthly_income,
        user_input.monthly_expense_total,
        user_input.savings_rate,
        user_input.debt_to_income_ratio,
        user_input.credit_score,
        user_input.loan_payment,
        user_input.investment_amount,
        user_input.discretionary_spending,
        user_input.essential_spending,
        user_input.emergency_fund
    ]])
    rf_scaled = rf_scaler.transform(rf_input)
    cash_flow = le.inverse_transform(rf.predict(rf_scaled))[0]

    return {"segment": segment, "cash_flow_status": cash_flow}

# =====================================================
# ROUTES
# =====================================================

@app.get("/")
def root():
    return {"message": "Finance Advisor AI is running."}


@app.post("/analyze")
def analyze(user_input: UserFinanceInput):
    # Step 1: ML prediction
    ml_result = predict(user_input)
    segment = ml_result["segment"]
    cash_flow = ml_result["cash_flow_status"]

    # Step 2: Build RAG query from user profile
    rag_query = (
        f"User is in segment '{segment}' with {cash_flow} cash flow. "
        f"Debt-to-income ratio is {user_input.debt_to_income_ratio}. "
        f"Savings rate is {user_input.savings_rate}. "
        f"Question: {user_input.question}"
    )
    context = get_context(rag_query, n_results=3)

    # Step 3: Build prompt for Claude
    prompt = f"""You are a personal finance advisor AI. Use the financial profile and reference knowledge below to give clear, actionable advice.

## User Financial Profile
- Monthly Income: ${user_input.monthly_income:,.0f}
- Monthly Expenses: ${user_input.monthly_expense_total:,.0f}
- Savings Rate: {user_input.savings_rate * 100:.1f}%
- Debt-to-Income Ratio: {user_input.debt_to_income_ratio:.2f}
- Credit Score: {user_input.credit_score:.0f}
- Emergency Fund: ${user_input.emergency_fund:,.0f}
- Investment Amount: ${user_input.investment_amount:,.0f}

## ML Analysis Results
- Financial Segment: {segment}
- Cash Flow Status: {cash_flow}

## Reference Knowledge
{context}

## User Question
{user_input.question}

## Instructions
- Give 3-5 specific, prioritized action steps based on this user's actual numbers
- Reference their segment and cash flow status in your response
- Be direct and concise, avoid generic advice
- Use dollar amounts and percentages where relevant
"""

    # Step 4: Call Claude API
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    advice = message.content[0].text

    return {
        "segment": segment,
        "cash_flow_status": cash_flow,
        "advice": advice
    }
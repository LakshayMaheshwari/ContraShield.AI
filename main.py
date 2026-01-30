from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI(title="ContractShield AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AnalyzeRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok", "message": "ContractShield AI backend is running"}

@app.post("/analyze")
def analyze_contract(req: AnalyzeRequest):
    contract_text = req.text.strip()

    if len(contract_text) < 30:
        return {"error": "Please paste a longer contract text."}

    prompt = f"""
You are ContractShield AI, an expert contract risk analyzer.

Return ONLY valid JSON:
{{
  "summary": ["..."],
  "risk_flags": [
    {{
      "title": "...",
      "severity": "RED|YELLOW|GREEN",
      "reason": "...",
      "suggested_fix": "..."
    }}
  ],
  "negotiation_points": ["..."]
}}

Contract:
\"\"\"{contract_text}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No extra text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return {"result": response.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}

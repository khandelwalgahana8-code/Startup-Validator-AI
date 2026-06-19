from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Startup Validator AI")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq Client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Input Model
class IdeaInput(BaseModel):
    idea: str


@app.get("/")
def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")

    return {
        "message": "Startup Validator AI Backend Running"
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "service": "Startup Validator AI"
    }


@app.post("/validate")
def validate(data: IdeaInput):

    prompt = f"""
You are a world-class startup investor, venture capitalist,
McKinsey consultant, Shark Tank judge and startup strategist.

Analyze the startup idea below.

STARTUP IDEA:
{data.idea}

Return ONLY RAW JSON.

Do NOT use markdown.
Do NOT use ```json.
Do NOT add explanations.
Do NOT add text before or after the JSON.

JSON FORMAT:

{{
    "startup_name": "",
    "executive_summary": "",

    "scores": {{
        "innovation": 0,
        "market_demand": 0,
        "revenue_potential": 0,
        "funding_potential": 0,
        "scalability": 0,
        "risk": 0,
        "overall": 0
    }},

    "strengths": [],
    "weaknesses": [],
    "opportunities": [],
    "threats": [],

    "target_customers": "",
    "market_size": "",
    "revenue_model": "",
    "competitors": [],

    "development_cost": "",
    "investment_required": "",

    "business_model": "",
    "go_to_market_strategy": "",

    "investor_verdict": "",
    "growth_potential": "",
    "recommendation": "",

    "success_probability": 0,
    "valuation_estimate": ""
}}
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=3000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        result = response.choices[0].message.content.strip()

        # Remove markdown wrappers
        result = result.replace("```json", "")
        result = result.replace("```JSON", "")
        result = result.replace("```", "")
        result = result.strip()

        # Extract JSON safely
        start = result.find("{")
        end = result.rfind("}") + 1

        if start != -1 and end != -1:
            result = result[start:end]

        parsed = json.loads(result)

        return {
            "success": True,
            "data": parsed
        }

    except json.JSONDecodeError as e:

        return {
            "success": False,
            "error": f"JSON Parse Error: {str(e)}",
            "raw_response": result if "result" in locals() else None
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(title="Code Interpreter API")


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic Models
# ============================================================

class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# ============================================================
# Python Code Execution Tool
# ============================================================

def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return the exact execution result.

    Returns:
        {
            "success": bool,
            "output": str
        }
    """

    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)

        output = sys.stdout.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:
        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:
        sys.stdout = old_stdout


# ============================================================
# AI Error Analysis
# ============================================================

def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    """
    Ask the LLM to identify the line number(s) responsible
    for the Python error.

    Returns:
        List[int]
    """

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://aipipe.org/openrouter/v1"
    )

    prompt = f"""
You are analyzing a Python execution failure.

Your job is ONLY to identify the exact source-code line number(s)
where the error occurred.

Use the traceback as the primary source of truth.
Do not invent line numbers.
Do not include traceback line numbers that refer only to Python's
internal execution machinery.

CODE:
{code}

TRACEBACK:
{error_traceback}

Return ONLY valid JSON in exactly this format:

{{
    "error_lines": [3]
}}

If there are multiple relevant source-code error lines, include them
as integers in the list.
"""

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash-lite",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_object"
        }
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("AI returned an empty response.")

    analysis = ErrorAnalysis.model_validate_json(content)

    return analysis.error_lines


# ============================================================
# Main Endpoint
# ============================================================

@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):

    # --------------------------------------------------------
    # 1. Execute Python code
    # --------------------------------------------------------

    execution = execute_python_code(request.code)

    # --------------------------------------------------------
    # 2. Successful execution
    # --------------------------------------------------------

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # --------------------------------------------------------
    # 3. Error occurred -> call AI
    # --------------------------------------------------------

    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    # --------------------------------------------------------
    # 4. Return AI analysis + EXACT traceback
    # --------------------------------------------------------

    return {
        "error": error_lines,
        "result": execution["output"]
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Code Interpreter API is running"
    }
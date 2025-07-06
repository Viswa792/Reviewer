# app.py (Final Production Version - Improved Prompt and Resilience)
import json
import os
import requests
import time
import docx
import threading
import traceback
import zipfile
import google.generativeai as genai
import streamlit as st
from queue import Queue
from urllib.parse import urlparse
from dotenv import load_dotenv

# --- CONFIGURATION & MODELS ---
load_dotenv()
REVIEW_MODELS = ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.5-flash-latest"]
VALIDATION_MODEL = "gemini-1.5-pro-latest"
RATE_LIMIT = 5

# --- DATA DOWNLOADER CLASS ---
class DataDownloader:
    def __init__(self, auth_token):
        self.auth_token = auth_token

    def download_zip_file(self, url, save_path):
        try:
            clean_auth_token = self.auth_token.strip()
            headers = {"Authorization": f"Bearer {clean_auth_token}"}
            clean_url = url.strip()
            response = requests.get(clean_url, headers=headers, stream=True, timeout=120)
            response.raise_for_status()
            task_id_match = urlparse(clean_url).query
            task_id = dict(param.split('=') for param in task_id_match.split('&')).get('ids[]')
            filename = f"task_{task_id}.zip" if task_id else "downloaded_notebook.zip"
            full_path = os.path.join(save_path, filename)
            os.makedirs(save_path, exist_ok=True)
            with open(full_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            return full_path
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"A network error occurred during download: {e}")

# --- UTILITY AND LOGIC FUNCTIONS ---
def find_and_unzip(zip_path, extract_folder):
    os.makedirs(extract_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    for root, _, files in os.walk(extract_folder):
        for file in files:
            if file.endswith(".ipynb"):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No .ipynb file found in the extracted content of {zip_path}")

def call_gemini_api(prompt, system_prompt=None, model=VALIDATION_MODEL):
    time.sleep(RATE_LIMIT)
    safety_settings = {'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE','HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE','HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE','HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'}
    generation_config = genai.types.GenerationConfig(
        temperature=0.2, 
        max_output_tokens=8192, 
        response_mime_type="application/json"
    )
    model_obj = genai.GenerativeModel(model_name=model, safety_settings=safety_settings, generation_config=generation_config, system_instruction=system_prompt)
    response = model_obj.generate_content(prompt)
    return response.text

def load_notebook_cells(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f).get('cells', [])

def preprocess_notebook(cells):
    structured, turn_counter = {}, 0
    for i, cell in enumerate(cells):
        source, cell_num, role = "".join(cell.get('source', [])), i + 1, "unknown"
        markers = {"**[system]**": "system_prompt", "**[user]**": "user_query", "**[assistant]**": "assistant_response", "**[thinking]**": "assistant_thinking", "**[thought]**": "assistant_thought", "**[tool_use]**": "tool_code", "**[tool_output]**": "tool_output", "**[tools]**": "tool_definitions"}
        for marker, r in markers.items():
            if marker in source: role = r; break
        if role == "user_query":
            turn_counter += 1
            structured[f"turn_{turn_counter}"] = []
        if role != "unknown":
            if f"turn_{turn_counter}" not in structured:
                if turn_counter == 0: turn_counter = 1
                structured[f"turn_{turn_counter}"] = []
            structured[f"turn_{turn_counter}"].append({"cell_number": cell_num, "role": role, "content": source})
    return json.dumps(structured, indent=2)

def load_guidelines(guidelines_dir):
    guidelines = {}
    for filename in sorted(os.listdir(guidelines_dir)):
        if filename.endswith(".docx") and not filename.startswith("~$"):
            file_path = os.path.join(guidelines_dir, filename)
            doc = docx.Document(file_path)
            guidelines[os.path.splitext(filename)[0]] = "\n".join(para.text for para in doc.paragraphs)
    return guidelines

def extract_json_from_response(response_text):
    if not response_text: raise ValueError("API response was empty.")
    if response_text.strip().startswith("```json"): response_text = response_text.strip()[7:-3]
    elif response_text.strip().startswith("```"): response_text = response_text.strip()[3:-3]
    json_match = response_text[response_text.find('{'):response_text.rfind('}') + 1]
    if json_match: return json.loads(json_match)
    raise ValueError(f"JSON Parsing Failed. The model returned conversational text instead of JSON. Raw Text: {response_text}")

# --- PROMPT ENGINEERING & REPORTING ---

# --- 1. THE NEW, IMPROVED PROMPT ---
def build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content):
    """Builds a heavily revised prompt with strong instructions and few-shot examples."""
    return f"""Your task is to act as a specialized AI auditor. You will analyze a Jupyter Notebook against a single, specific guideline.
Your response MUST be a single, valid JSON object and NOTHING ELSE. Do not include conversational text, apologies, or any text outside the final JSON object.

**Your Sole Focus: The "{guideline_name}" Guideline**
---
{guideline_content}
---

**Mandatory Instructions:**
1.  Carefully analyze the entire notebook provided below.
2.  For each violation of the `{guideline_name}` guideline, provide your step-by-step thinking process and a final, concise issue description.
3.  **Crucially**: If you find NO violations of the guideline, you MUST return an empty `findings` array like this: `"findings": []`.
4.  Your response MUST conform to the JSON schema specified in the examples. Do not add, remove, or change keys.
5.  All strings within the JSON must have properly escaped characters (e.g., use `\\"` for a double quote).

**Output Examples:**

*Example 1: If you find violations*
```json
{{
  "guideline": "{guideline_name}",
  "findings": [
    {{
      "cell_number": 5,
      "thinking_process": "The user asked for the capital of France. The model responded with \\"The capital of France is Berlin.\\". This is a clear factual error, also known as a hallucination.",
      "issue_description": "In cell 5, the model incorrectly states that the capital of France is Berlin. The correct capital is Paris."
    }}
  ]
}}

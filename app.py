# app.py (with added debugging)
import json
import os
import requests
import time
import docx
import threading
import traceback  # --- DEBUGGING CHANGE: Import traceback to get detailed error info
import zipfile
import google.generativeai as genai
import streamlit as st
from queue import Queue
from urllib.parse import urlparse
from dotenv import load_dotenv

# --- YOUR EXACT CONFIGURATION AND MODELS ---
load_dotenv()
REVIEW_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash"]
VALIDATION_MODEL = "gemini-2.5-pro"
RATE_LIMIT = 5

# --- DATA DOWNLOADER CLASS (UNCHANGED) ---
class DataDownloader:
    def __init__(self, auth_token, status_placeholder):
        self.auth_token = auth_token.strip()
        self.headers = {"Authorization": f"Bearer {self.auth_token}"}
        self.status_placeholder = status_placeholder

    def download_zip_file(self, url, save_path):
        clean_url = url.strip()
        try:
            response = requests.get(clean_url, headers=self.headers, stream=True, timeout=120)
            response.raise_for_status()
            task_id_match = urlparse(clean_url).query
            task_id = dict(param.split('=') for param in task_id_match.split('&')).get('ids[]')
            filename = f"task_{task_id}.zip" if task_id else "downloaded_notebook.zip"
            full_path = os.path.join(save_path, filename)
            os.makedirs(save_path, exist_ok=True)
            with open(full_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            self.status_placeholder.info(f"✅ Successfully downloaded ZIP file.")
            return full_path
        except requests.exceptions.RequestException as e:
            self.status_placeholder.error(f"❌ Error during download for URL '{clean_url}': {e}")
            return None

# --- ALL YOUR OTHER FUNCTIONS (find_and_unzip, API calls, prompts, etc.) ---
# These are kept exactly as they were in the last version, but adapted for Streamlit UI logging.

def find_and_unzip(zip_path, extract_folder, status_placeholder):
    # (Your logic, with st logging)
    try:
        os.makedirs(extract_folder, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        for root, _, files in os.walk(extract_folder):
            for file in files:
                if file.endswith(".ipynb"):
                    notebook_path = os.path.join(root, file)
                    status_placeholder.info(f"✅ Found notebook file: {os.path.basename(notebook_path)}")
                    return notebook_path
        status_placeholder.error(f"❌ No .ipynb file found in {zip_path}")
        return None
    except Exception as e:
        status_placeholder.error(f"❌ An error occurred during unzipping: {e}")
        return None

def call_gemini_api(prompt, system_prompt=None, model=VALIDATION_MODEL):
    # (Your logic)
    time.sleep(RATE_LIMIT)
    safety_settings = {
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
    }
    if "flash" in model:
        generation_config = genai.types.GenerationConfig(temperature=0.2, max_output_tokens=8192)
    else:
        generation_config = genai.types.GenerationConfig(
            temperature=0.2, max_output_tokens=8192, response_mime_type="application/json"
        )
    model_obj = genai.GenerativeModel(
        model_name=model,
        safety_settings=safety_settings,
        generation_config=generation_config,
        system_instruction=system_prompt
    )
    response = model_obj.generate_content(prompt)
    return response.text

# ... All your other helper functions like load_notebook_cells, preprocess_notebook, etc. go here ...
# They are omitted for brevity, but they should be in your actual file.
def load_notebook_cells(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f).get('cells', [])
    except Exception as e:
        st.error(f"❌ Error loading notebook: {e}")
        return None

def preprocess_notebook(cells):
    structured = {}
    turn_counter = 0
    for i, cell in enumerate(cells):
        source = "".join(cell.get('source', []))
        cell_num = i + 1
        role = "unknown"
        markers = {"**[system]**": "system_prompt", "**[user]**": "user_query", "**[assistant]**": "assistant_response", "**[thinking]**": "assistant_thinking", "**[thought]**": "assistant_thought", "**[tool_use]**": "tool_code", "**[tool_output]**": "tool_output", "**[tools]**": "tool_definitions"}
        for marker, r in markers.items():
            if marker in source:
                role = r
                break
        if role == "user_query":
            turn_counter += 1
            structured[f"turn_{turn_counter}"] = []
        if role != "unknown":
            if f"turn_{turn_counter}" not in structured:
                if turn_counter == 0: turn_counter = 1
                structured[f"turn_{turn_counter}"] = []
            structured[f"turn_{turn_counter}"].append({"cell_number": cell_num, "role": role, "content": source})
    return json.dumps(structured, indent=2)

def load_guidelines(guidelines_dir, status_placeholder):
    guidelines = {}
    try:
        for filename in sorted(os.listdir(guidelines_dir)):
            if filename.endswith(".docx") and not filename.startswith("~$"):
                file_path = os.path.join(guidelines_dir, filename)
                doc = docx.Document(file_path)
                content = "\n".join(para.text for para in doc.paragraphs)
                guideline_name = os.path.splitext(filename)[0]
                guidelines[guideline_name] = content
        status_placeholder.info(f"✅ Loaded guidelines: {list(guidelines.keys())}")
        return guidelines
    except Exception as e:
        status_placeholder.error(f"❌ Error loading guidelines: {e}")
        return None

def extract_json_from_response(response_text):
    if not response_text: return None
    try:
        if response_text.strip().startswith("```json"): response_text = response_text.strip()[7:-3]
        elif response_text.strip().startswith("```"): response_text = response_text.strip()[3:-3]
        json_match = response_text[response_text.find('{'):response_text.rfind('}') + 1]
        if json_match: return json.loads(json_match)
    except (json.JSONDecodeError, IndexError) as e:
        # This error is now critical to see
        raise Exception(f"JSON Parsing Failed: {e}. Raw Text: {response_text}")

# --- PROMPT ENGINEERING & REPORTING FUNCTIONS (UNCHANGED) ---
def build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content):
    # Your exact prompt logic
    return f"..."
def build_validation_prompt(structured_notebook, all_findings):
    # Your exact prompt logic
    return f"..."
def generate_final_report(validation_result, notebook_name):
    # Your exact reporting logic
    return "..."

# --- WORKER FUNCTION FOR THREADING (MODIFIED FOR DEBUGGING) ---
def run_review_task(queue, structured_notebook, model_name, guideline_name, guideline_content, status_placeholder):
    try:  # --- DEBUGGING CHANGE: Wrap the entire worker function in a try...except block
        status_placeholder.info(f"  - ⏳ Assigning '{guideline_name}' review to {model_name}...")
        system_prompt = "You are a meticulous AI Notebook Auditor..." # Your prompt
        prompt = build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content)

        # This call is now inside the master try...except block
        response = call_gemini_api(prompt, system_prompt=system_prompt, model=model_name)

        if response:
            findings_data = extract_json_from_response(response)
            if findings_data and "findings" in findings_data and isinstance(findings_data.get("findings"), list):
                findings = findings_data["findings"]
                for finding in findings:
                    finding["auditor"] = model_name
                    finding["guideline"] = guideline_name
                queue.put(findings) # Put successful findings on the queue
                status_placeholder.info(f"  - ✅ {model_name} completed '{guideline_name}' review, found {len(findings)} potential issues.")
                return

        # If we reach here, the response was bad
        raise Exception("Review failed, API response was empty or malformed.")

    except Exception as e:
        # --- DEBUGGING CHANGE: If ANY error occurs, put a detailed error object on the queue
        error_details = {
            "error": True,
            "guideline": guideline_name,
            "model": model_name,
            "message": str(e),
            "traceback": traceback.format_exc()  # The full, detailed traceback
        }
        queue.put(error_details)


# --- MAIN WORKFLOW (MODIFIED FOR DEBUGGING) ---
def run_audit_

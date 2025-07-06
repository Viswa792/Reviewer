# app.py (Final, Aggressive Debugging Version)
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

# --- YOUR EXACT CONFIGURATION AND MODELS ---
load_dotenv()
REVIEW_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash"]
VALIDATION_MODEL = "gemini-2.5-pro"
RATE_LIMIT = 5

# --- DATA DOWNLOADER CLASS (MODIFIED FOR AGGRESSIVE DEBUGGING) ---
class DataDownloader:
    def __init__(self, auth_token, status_placeholder):
        self.auth_token = auth_token
        self.status_placeholder = status_placeholder

    def download_zip_file(self, url, save_path):
        # --- NEW DEBUGGING BLOCK ---
        self.status_placeholder.info("--- Entering Download Step: Aggressive Debugging ---")
        self.status_placeholder.info(f"Auth Token Type: {type(self.auth_token)}")
        self.status_placeholder.info(f"Auth Token Representation (repr): {repr(self.auth_token)}")
        
        # Clean the token right before use
        clean_auth_token = self.auth_token.strip()
        headers = {"Authorization": f"Bearer {clean_auth_token}"}
        
        self.status_placeholder.info(f"Header being sent: {{'Authorization': 'Bearer {clean_auth_token[:5]}...'}}")

        self.status_placeholder.info(f"URL Type as received: {type(url)}")
        self.status_placeholder.info(f"URL Representation (repr) as received: {repr(url)}")
        
        clean_url = url.strip()
        
        self.status_placeholder.info(f"Cleaned URL Representation (repr): {repr(clean_url)}")
        self.status_placeholder.info("--- Attempting Download Now ---")
        # --- END OF NEW DEBUGGING BLOCK ---
        
        try:
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
            self.status_placeholder.info(f"✅ Successfully downloaded ZIP file.")
            return full_path
        except requests.exceptions.RequestException as e:
            self.status_placeholder.error(f"❌ Error during download for URL '{clean_url}': {e}")
            return None

# --- ALL OTHER FUNCTIONS (UNCHANGED FROM PREVIOUS DEBUG VERSION) ---
def find_and_unzip(zip_path, extract_folder, status_placeholder):
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
    time.sleep(RATE_LIMIT)
    safety_settings = {'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE','HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE','HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE','HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',}
    if "flash" in model:
        generation_config = genai.types.GenerationConfig(temperature=0.2, max_output_tokens=8192)
    else:
        generation_config = genai.types.GenerationConfig(temperature=0.2, max_output_tokens=8192, response_mime_type="application/json")
    model_obj = genai.GenerativeModel(model_name=model,safety_settings=safety_settings,generation_config=generation_config,system_instruction=system_prompt)
    response = model_obj.generate_content(prompt)
    return response.text

def load_notebook_cells(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f).get('cells', [])
    except Exception as e:
        st.error(f"❌ Error loading notebook: {e}")
        return None

def preprocess_notebook(cells):
    structured, turn_counter = {}, 0
    for i, cell in enumerate(cells):
        source, cell_num, role = "".join(cell.get('source', [])), i + 1, "unknown"
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
        raise Exception(f"JSON Parsing Failed: {e}. Raw Text: {response_text}")

def build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content):
    return f"""...""" # Your prompt logic
def build_validation_prompt(structured_notebook, all_findings):
    return f"""...""" # Your prompt logic
def generate_final_report(validation_result, notebook_name):
    return "..." # Your report logic

def run_review_task(queue, structured_notebook, model_name, guideline_name, guideline_content, status_placeholder):
    try:
        status_placeholder.info(f"  - ⏳ Assigning '{guideline_name}' review to {model_name}...")
        system_prompt = "You are a meticulous AI Notebook Auditor..."
        prompt = build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content)
        response = call_gemini_api(prompt, system_prompt=system_prompt, model=model_name)
        if response:
            findings_data = extract_json_from_response(response)
            if findings_data and "findings" in findings_data and isinstance(findings_data.get("findings"), list):
                queue.put(findings_data["findings"])
                status_placeholder.info(f"  - ✅ {model_name} completed '{guideline_name}'.")
                return
        raise Exception("Review failed: API response was empty or malformed.")
    except Exception as e:
        queue.put({"error": True,"guideline": guideline_name,"model": model_name,"message": str(e),"traceback": traceback.format_exc()})

def run_audit_workflow(task_number, status_placeholder):
    errors_found = []
    try:
        status_placeholder.info("🚀 Audit initiated...")
        AUTH_TOKEN = st.secrets["AUTH_TOKEN"]
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=GEMINI_API_KEY.strip())

        status_placeholder.info(f"[1/7] Downloading notebook...")
        notebook_zip_url = f"https://labeling-s.turing.com/api/conversations/download-notebook-zip?ids[]={task_number}"
        download_dir = "./downloaded_notebooks"
        downloader = DataDownloader(AUTH_TOKEN, status_placeholder)
        zip_path = downloader.download_zip_file(notebook_zip_url, download_dir)
        if not zip_path: return None, None, [{"error": True, "guideline": "Download", "message": "Download failed. Check logs."}]

        # ... Rest of the workflow ...
        return "Report", "filename.md", []
    
    except Exception as e:
        errors_found.append({"error": True, "guideline": "Main Workflow", "message": str(e), "traceback": traceback.format_exc()})
        return None, None, errors_found

if __name__ == "__main__":
    st.set_page_config(page_title="AI Notebook Auditor [Debug Mode]", layout="wide")
    st.title("🤖 AI Notebook Auditor [Debug Mode]")
    st.markdown("This version will perform aggressive debugging on the download step.")
    task_number = st.text_input("Enter the Task Number:", placeholder="e.g., 214514")
    if st.button("Start Review", type="primary", use_container_width=True):
        if task_number and task_number.isdigit():
            console_container = st.expander("Live Console Log", expanded=True)
            status_placeholder = console_container.empty()
            with st.spinner("Executing audit..."):
                final_report_md, report_filename, errors = run_audit_workflow(task_number, status_placeholder)
            if errors:
                st.error("One or more tasks failed! Here are the details:")
                for error in errors:
                    st.subheader(f"Error in step: `{error['guideline']}`")
                    st.write(f"**Error Message:** {error.get('message', 'N/A')}")
            elif final_report_md:
                st.balloons()
                st.header("Generated Review Content")
        else:
            st.error("❗ Please enter a valid, numeric task number.")

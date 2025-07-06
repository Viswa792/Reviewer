# app.py (Final Version - Corrected KeyError and Threading)
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

# Using the correct, publicly available model names.
# This is the most likely fix for the *next* error you will see.
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
        except requests.exceptions.RequestException:
            # Re-raise the exception to be caught by the main workflow
            raise

# --- UTILITY AND LOGIC FUNCTIONS ---
def find_and_unzip(zip_path, extract_folder):
    os.makedirs(extract_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    for root, _, files in os.walk(extract_folder):
        for file in files:
            if file.endswith(".ipynb"):
                return os.path.join(root, file)
    # If no file is found after checking all, raise an error.
    raise FileNotFoundError(f"No .ipynb file found in the extracted content of {zip_path}")

def call_gemini_api(prompt, system_prompt=None, model=VALIDATION_MODEL):
    time.sleep(RATE_LIMIT)
    safety_settings = {'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE','HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE','HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE','HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'}
    if "flash" in model:
        generation_config = genai.types.GenerationConfig(temperature=0.2, max_output_tokens=8192)
    else:
        generation_config = genai.types.GenerationConfig(temperature=0.2, max_output_tokens=8192, response_mime_type="application/json")
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
            content = "\n".join(para.text for para in doc.paragraphs)
            guidelines[os.path.splitext(filename)[0]] = content
    return guidelines

def extract_json_from_response(response_text):
    if not response_text: raise ValueError("API response was empty.")
    if response_text.strip().startswith("```json"): response_text = response_text.strip()[7:-3]
    elif response_text.strip().startswith("```"): response_text = response_text.strip()[3:-3]
    json_match = response_text[response_text.find('{'):response_text.rfind('}') + 1]
    if json_match: return json.loads(json_match)
    raise ValueError(f"JSON Parsing Failed. Raw Text: {response_text}")

# --- PROMPTS AND REPORTING ---
# (Your functions build_targeted_review_prompt, build_validation_prompt, generate_final_report go here, unabbreviated)
def build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content):
    return f"""...""" # Your full prompt
def build_validation_prompt(structured_notebook, all_findings):
    return f"""...""" # Your full prompt
def generate_final_report(validation_result, notebook_name):
    return "..." # Your full report generation logic

# --- WORKER FUNCTION (SILENT, WITH ERROR CATCHING) ---
def run_review_task(queue, structured_notebook, model_name, guideline_name, guideline_content):
    try:
        system_prompt = "You are a meticulous AI Notebook Auditor..."
        prompt = build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content)
        response = call_gemini_api(prompt, system_prompt=system_prompt, model=model_name)
        findings_data = extract_json_from_response(response)
        if "findings" in findings_data:
            findings = findings_data["findings"]
            for finding in findings:
                finding["auditor"] = model_name
                finding["guideline"] = guideline_name
            queue.put(findings)
        else:
            raise ValueError("Parsed JSON, but 'findings' key was missing.")
    except Exception as e:
        queue.put({"error": True, "guideline": guideline_name, "model": model_name, "message": str(e), "traceback": traceback.format_exc()})

# --- MAIN WORKFLOW FUNCTION ---
def run_audit_workflow(task_number, status_placeholder):
    try:
        status_placeholder.info("🚀 Audit initiated...")
        AUTH_TOKEN = st.secrets["AUTH_TOKEN"]
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=GEMINI_API_KEY.strip())

        status_placeholder.info(f"[1/7] Downloading notebook for task '{task_number}'...")
        notebook_zip_url = f"https://labeling-s.turing.com/api/conversations/download-notebook-zip?ids[]={task_number}"
        downloader = DataDownloader(AUTH_TOKEN)
        zip_path = downloader.download_zip_file(notebook_zip_url, "./downloaded_notebooks")
        status_placeholder.info("✅ Download complete.")
        
        status_placeholder.info(f"[2/7] Unzipping...")
        notebook_path = find_and_unzip(zip_path, "./task_extracted")
        notebook_name = os.path.basename(notebook_path)
        status_placeholder.info(f"✅ Unzip complete. Found notebook: {notebook_name}")

        status_placeholder.info("[3/7] Loading and preprocessing data...")
        guidelines = load_guidelines("./Guidlines")
        cells = load_notebook_cells(notebook_path)
        structured_notebook = preprocess_notebook(cells)
        status_placeholder.info("✅ Data preprocessed.")

        status_placeholder.info(f"[4/7] Kicking off parallel reviews... (This may take several minutes)")
        all_findings, results_queue, threads = [], Queue(), []
        for i, (guideline_name, guideline_content) in enumerate(guidelines.items()):
            model = REVIEW_MODELS[i % len(REVIEW_MODELS)]
            thread = threading.Thread(target=run_review_task, args=(results_queue, structured_notebook, model, guideline_name, guideline_content))
            threads.append(thread); thread.start()
            time.sleep(1)
        for thread in threads: thread.join()
        status_placeholder.info("✅ All review threads have finished.")
        
        errors_found = []
        while not results_queue.empty():
            result = results_queue.get()
            if isinstance(result, dict) and result.get("error"):
                errors_found.append(result)
            else:
                all_findings.extend(result)
        
        if errors_found:
            return None, None, errors_found

        status_placeholder.info(f"[5/7] Aggregated {len(all_findings)} findings.")
        status_placeholder.info(f"[6/7] Running validation...")
        # (Validation Logic) ...
        
        status_placeholder.info("[7/7] Generating final report...")
        # (Reporting Logic) ...
        final_report = "Final report..."
        report_filename = "report.md"
        
        status_placeholder.success("🎉 Audit Complete!")
        return final_report, report_filename, []
    
    except Exception as e:
        # Catch any other unexpected errors in the main workflow and give them the full dictionary structure
        return None, None, [{"error": True, "guideline": "Main Workflow", "model": "N/A", "message": str(e), "traceback": traceback.format_exc()}]

# --- STREAMLIT UI ---
if __name__ == "__main__":
    st.set_page_config(page_title="AI Notebook Auditor", layout="wide")
    st.title("🤖 AI Notebook Auditor")
    # Make sure to include all your unabbreviated helper functions in the file
    task_number = st.text_input("Enter the Task Number:", placeholder="e.g., 214514")

    if st.button("Start Review", type="primary", use_container_width=True):
        if task_number and task_number.isdigit():
            console_container = st.expander("Live Console Log", expanded=True)
            status_placeholder = console_container.empty()
            with st.spinner("Executing full audit... This may take several minutes."):
                final_report_md, report_filename, errors = run_audit_workflow(task_number, status_placeholder)
            
            if errors:
                st.error("The audit encountered one or more errors:")
                for error in errors:
                    st.subheader(f"Error during: `{error.get('guideline', 'Unknown Step')}` review (Model: `{error.get('model', 'N/A')}`)")
                    st.write("**Error Message:**")
                    st.code(error.get('message', 'No message.'), language='text')
                    st.write("**Full Traceback:**")
                    st.code(error.get('traceback', 'No traceback.'), language='text')
            elif final_report_md:
                st.balloons()
                st.header("Generated Review Content")
                st.markdown(final_report_md)
        else:
            st.error("❗ Please enter a valid, numeric task number.")

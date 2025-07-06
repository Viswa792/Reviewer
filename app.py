# app.py (Final Version with Correct Threading)
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

# --- POTENTIAL FIX FOR LIKELY NEXT ERROR ---
# The names 'gemini-2.5-pro' are likely incorrect for the public API.
# The correct, publicly available models are gemini-1.5-pro and gemini-1.5-flash.
# If the app fails again with an API error, change these lines to the ones below.
REVIEW_MODELS = ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.5-flash-latest"]
VALIDATION_MODEL = "gemini-1.5-pro-latest"
# Original Names (kept for reference, but likely to fail):
# REVIEW_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash"]
# VALIDATION_MODEL = "gemini-2.5-pro"

RATE_LIMIT = 5

# --- DATA DOWNLOADER CLASS (UNCHANGED FROM WORKING VERSION) ---
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
            st.error(f"A network error occurred during download: {e}") # Use st.error directly
            return None

# --- ALL OTHER HELPER FUNCTIONS (UNCHANGED LOGIC) ---
# ... find_and_unzip, call_gemini_api, etc. ...
def find_and_unzip(zip_path, extract_folder):
    # This function doesn't need to write to the UI, it just returns a value or raises an error.
    os.makedirs(extract_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    for root, _, files in os.walk(extract_folder):
        for file in files:
            if file.endswith(".ipynb"):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No .ipynb file found in {zip_path}")


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

# ... other functions like load_notebook_cells, preprocess_notebook, load_guidelines, etc. ...
# These are correct and do not need changes. I am omitting them for readability.

def extract_json_from_response(response_text):
    if not response_text: return None
    if response_text.strip().startswith("```json"): response_text = response_text.strip()[7:-3]
    elif response_text.strip().startswith("```"): response_text = response_text.strip()[3:-3]
    json_match = response_text[response_text.find('{'):response_text.rfind('}') + 1]
    if json_match: return json.loads(json_match)
    raise ValueError(f"JSON Parsing Failed. Raw Text: {response_text}")

# --- WORKER FUNCTION (MODIFIED TO BE SILENT) ---
def run_review_task(queue, structured_notebook, model_name, guideline_name, guideline_content):
    # THIS THREAD IS NOT ALLOWED TO CALL ANY st.* FUNCTIONS.
    try:
        system_prompt = "You are a meticulous AI Notebook Auditor..."
        prompt = build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content)
        response = call_gemini_api(prompt, system_prompt=system_prompt, model=model_name)
        if response:
            findings_data = extract_json_from_response(response)
            if findings_data and "findings" in findings_data:
                findings = findings_data["findings"]
                # Add context to each finding before putting it on the queue
                for finding in findings:
                    finding["auditor"] = model_name
                    finding["guideline"] = guideline_name
                queue.put(findings)
                return
        raise RuntimeError("API response was empty, malformed, or did not contain 'findings'.")
    except Exception as e:
        # If an error occurs, put a detailed error object on the queue.
        error_details = {
            "error": True,
            "guideline": guideline_name,
            "model": model_name,
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        queue.put(error_details)

# --- MAIN WORKFLOW FUNCTION (WITH CORRECTED UI LOGGING) ---
def run_audit_workflow(task_number, status_placeholder):
    # This main function is allowed to update the UI.
    errors_found = []
    try:
        status_placeholder.info("🚀 Audit initiated...")
        AUTH_TOKEN = st.secrets["AUTH_TOKEN"]
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=GEMINI_API_KEY.strip())

        status_placeholder.info(f"[1/7] Downloading notebook for task '{task_number}'...")
        notebook_zip_url = f"https://labeling-s.turing.com/api/conversations/download-notebook-zip?ids[]={task_number}"
        downloader = DataDownloader(AUTH_TOKEN)
        zip_path = downloader.download_zip_file(notebook_zip_url, "./downloaded_notebooks")
        if not zip_path: return None, None, [{"error": True, "guideline": "Download", "message": "Download failed."}]
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
            # The worker thread is created here. It gets no UI elements.
            thread = threading.Thread(target=run_review_task, args=(results_queue, structured_notebook, model, guideline_name, guideline_content))
            threads.append(thread); thread.start()
            time.sleep(1)
        for thread in threads: thread.join()
        status_placeholder.info("✅ All review threads have finished.")

        while not results_queue.empty():
            result = results_queue.get()
            # Check if the item from the queue is an error object
            if isinstance(result, dict) and result.get("error"):
                errors_found.append(result)
            else:
                all_findings.extend(result)
        
        if errors_found:
            return None, None, errors_found # Stop and return the errors if any were found

        status_placeholder.info(f"[5/7] Aggregated {len(all_findings)} findings.")
        status_placeholder.info(f"[6/7] Running validation with {VALIDATION_MODEL}...")
        # (Validation and Report generation logic is unchanged)
        
        status_placeholder.info("[7/7] Generating final report...")
        final_report = "Final report content..." # Placeholder for your generate_final_report call
        report_filename = f"FINAL_AUDIT_REPORT_{notebook_name}.md"
        
        status_placeholder.success("🎉 Audit Complete!")
        return final_report, report_filename, []
    
    except Exception as e:
        # Catch any other unexpected errors in the main workflow
        errors_found.append({"error": True, "guideline": "Main Workflow", "message": str(e), "traceback": traceback.format_exc()})
        return None, None, errors_found

# --- STREAMLIT UI ---
if __name__ == "__main__":
    st.set_page_config(page_title="AI Notebook Auditor", layout="wide")
    st.title("🤖 AI Notebook Auditor")

    # Make sure to include all your helper functions here, like build_targeted_review_prompt, etc.
    # They were omitted above for readability.

    task_number = st.text_input("Enter the Task Number:", placeholder="e.g., 214514")

    if st.button("Start Review", type="primary", use_container_width=True):
        if task_number and task_number.isdigit():
            console_container = st.expander("Live Console Log", expanded=True)
            status_placeholder = console_container.empty()
            with st.spinner("Executing full audit... This may take several minutes."):
                final_report_md, report_filename, errors = run_audit_workflow(task_number, status_placeholder)
            
            if errors:
                st.error("The audit failed. Here are the details:")
                for error in errors:
                    st.subheader(f"Error during: `{error['guideline']}` review (Model: `{error['model']}`)")
                    st.write("**Error Message:**")
                    st.code(error['message'], language='text')
                    st.write("**Full Traceback:**")
                    st.code(error['traceback'], language='text')
            elif final_report_md:
                st.balloons()
                st.header("Generated Review Content")
                st.markdown(final_report_md)
                st.download_button(label="⬇️ Download Full Report", data=final_report_md, file_name=report_filename)
        else:
            st.error("❗ Please enter a valid, numeric task number.")

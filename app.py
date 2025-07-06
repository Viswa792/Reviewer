# app.py (Final Debug Version - Complete and Unabbreviated)
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

# --- DATA DOWNLOADER CLASS (ADAPTED FOR STREAMLIT) ---
class DataDownloader:
    def __init__(self, auth_token, status_placeholder):
        if not auth_token:
            raise ValueError("Authentication token is missing.")
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

# --- UTILITY FUNCTIONS (ADAPTED FOR STREAMLIT) ---
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

# --- API CALL FUNCTION (UNCHANGED LOGIC) ---
def call_gemini_api(prompt, system_prompt=None, model=VALIDATION_MODEL):
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

# --- FILE HANDLING & PREPROCESSING (UNCHANGED LOGIC) ---
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
        raise Exception(f"JSON Parsing Failed: {e}. Raw Text: {response_text}")

# --- PROMPT ENGINEERING & REPORTING FUNCTIONS (YOUR EXACT CODE) ---
def build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content):
    return f"""As an expert AI auditor specializing in the '{guideline_name}' guideline, your task is to perform a focused review of the provided Jupyter Notebook.
**Your Sole Focus: "{guideline_name}" Guideline**
---
{guideline_content}
---
**Instructions:**
1.  Analyze the entire notebook provided below with extreme attention to detail.
2.  For each potential violation, first provide your step-by-step thinking process.
3.  Based on your thinking, write a final, concise issue description.
4.  **IMPORTANT**: When writing the 'thinking_process' or 'issue_description', ensure that any double quotes (") within the text are properly escaped with a backslash (\\").
5.  If you find no violations, return an empty "findings" array.
6.  Your response MUST be a single, valid JSON object and NOTHING ELSE. Do not include markdown formatting like ```json or any text whatsoever outside of the curly braces of the JSON object.
**Notebook to Analyze:**
---
{structured_notebook}
---
**Required Output Format (JSON only):**
```json
{{
  "guideline": "{guideline_name}",
  "findings": [
    {{
      "cell_number": <integer>,
      "thinking_process": "<Think step-by-step. Remember to escape internal double quotes like this: \\". This is critical.>",
      "issue_description": "<Describe the issue. Remember to escape internal double quotes like this: \\".>"
    }}
  ]
}}
```"""

def build_validation_prompt(structured_notebook, all_findings):
    return f"""As the Senior AI Audit Validator, your task is to review all preliminary findings from a team of specialized AI auditors, validate them, and compile a final, clean report.
**Instructions:**
1.  Review all findings provided below. Each finding is marked with its 'guideline' (e.g., 'Hallucination', 'logical').
2.  Validate each finding by cross-referencing it with the notebook structure and the auditor's thinking.
3.  Eliminate duplicate findings and incorrect "false positives."
4.  For each validated issue, provide a detailed explanation and your own validation reasoning.
5.  **Crucially, for each issue, you must include the `violation_category`, which should be the name of the guideline that was violated.**
6.  Organize all validated issues by their cell number.
7.  Provide a concise, high-level summary of the notebook's overall quality.
8.  Return your final validated report exclusively in a valid JSON format.
**Notebook to Analyze:**
---
{structured_notebook}
---
**All Findings to Validate (with auditor's thinking):**
---
{json.dumps(all_findings, indent=2)}
---
**Required Output Format (JSON only):**
```json
{{
  "final_report": [
    {{
      "cell_number": <integer>,
      "validated_issues": [
        {{
          "violation_category": "<The name of the guideline violated, e.g., 'Hallucination', 'logical', 'Structure'>",
          "problematic_content": "<Quote the exact problematic text or summarize the problematic action in the cell.>",
          "reason_for_violation": "<Explain exactly why this is a violation of the guidelines, referencing the specific rule if possible.>",
          "validation_reasoning": "<Explain your thought process for validating this issue. Did you agree with the initial auditor's thinking? Did you consolidate multiple findings? Explain why this is a confirmed violation.>"
        }}
      ]
    }}
  ],
  "overall_feedback": "<A brief, high-level summary of the audit results and the notebook's overall quality.>"
}}
```"""

def generate_final_report(validation_result, notebook_name):
    report = f"# Final Audit Report: {notebook_name}\n"
    report += f"**Generated at**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
    overall_feedback = validation_result.get("overall_feedback", "No overall feedback provided.")
    report += f"## Overall Feedback\n{overall_feedback}\n\n---\n\n"
    report += "## Cell-wise Findings\n\n"
    final_report_data = validation_result.get("final_report", [])
    if not final_report_data:
        report += "✅ No validated issues found in the final report.\n"
    else:
        sorted_report_data = sorted(final_report_data, key=lambda x: x.get("cell_number", float('inf')))
        for cell_data in sorted_report_data:
            cell_num = cell_data.get("cell_number", "N/A")
            issues = cell_data.get("validated_issues", [])
            if issues:
                report += f"### Cell {cell_num}\n"
                for issue in issues:
                    report += f"**Violation Category:** {issue.get('violation_category', 'N/A')}\n\n"
                    report += f"**Problematic Content:**\n```\n{issue.get('problematic_content', 'N/A')}\n```\n\n"
                    report += f"**Reason for Violation:**\n{issue.get('reason_for_violation', 'N/A')}\n\n"
                    report += f"**Validator's Reasoning:**\n*_{issue.get('validation_reasoning', 'N/A')}_*\n\n"
                report += "---\n"
    report += "\n*Report generated with a multi-model audit via Google Gemini.*"
    return report

# --- WORKER FUNCTION FOR THREADING (MODIFIED FOR DEBUGGING) ---
def run_review_task(queue, structured_notebook, model_name, guideline_name, guideline_content, status_placeholder):
    try:
        status_placeholder.info(f"  - ⏳ Assigning '{guideline_name}' review to {model_name}...")
        system_prompt = "You are a meticulous AI Notebook Auditor..."
        prompt = build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content)
        response = call_gemini_api(prompt, system_prompt=system_prompt, model=model_name)
        if response:
            findings_data = extract_json_from_response(response)
            if findings_data and "findings" in findings_data and isinstance(findings_data.get("findings"), list):
                findings = findings_data["findings"]
                for finding in findings:
                    finding["auditor"] = model_name
                    finding["guideline"] = guideline_name
                queue.put(findings)
                status_placeholder.info(f"  - ✅ {model_name} completed '{guideline_name}', found {len(findings)} issues.")
                return
        raise Exception("Review failed: API response was empty or malformed.")
    except Exception as e:
        error_details = {
            "error": True,
            "guideline": guideline_name,
            "model": model_name,
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        queue.put(error_details)

# --- MAIN WORKFLOW (MODIFIED FOR DEBUGGING) ---
def run_audit_workflow(task_number, status_placeholder):
    errors_found = []
    try:
        status_placeholder.info("🚀 Audit initiated...")
        AUTH_TOKEN = st.secrets["AUTH_TOKEN"].strip()
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"].strip()
        genai.configure(api_key=GEMINI_API_KEY)
        status_placeholder.info("✅ Gemini API configured.")

        status_placeholder.info(f"[1/7] Downloading notebook...")
        notebook_zip_url = f"[https://labeling-s.turing.com/api/conversations/download-notebook-zip?ids](https://labeling-s.turing.com/api/conversations/download-notebook-zip?ids)[]={task_number}"
        download_dir = "./downloaded_notebooks"
        downloader = DataDownloader(AUTH_TOKEN, status_placeholder)
        zip_path = downloader.download_zip_file(notebook_zip_url, download_dir)
        if not zip_path: return None, None, [{"error": True, "guideline": "Download", "model": "N/A", "message": "Download failed. Check logs above.", "traceback": "No traceback available."}]

        status_placeholder.info(f"[2/7] Unzipping notebook...")
        extract_path = os.path.join(download_dir, f"task_{task_number}_extracted")
        notebook_path = find_and_unzip(zip_path, extract_path, status_placeholder)
        if not notebook_path: return None, None, [{"error": True, "guideline": "Unzip", "model": "N/A", "message": "Failed to find .ipynb file.", "traceback": "No traceback available."}]
        notebook_name = os.path.basename(notebook_path)

        status_placeholder.info("[3/7] Preprocessing data...")
        guidelines_dir = "Guidlines"
        if not os.path.exists(guidelines_dir):
            raise Exception(f"CRITICAL: Guidelines directory '{guidelines_dir}' not found in repository.")
        cells = load_notebook_cells(notebook_path)
        if not cells: raise Exception("CRITICAL: Failed to load cells from notebook.")
        guidelines = load_guidelines(guidelines_dir, status_placeholder)
        if not guidelines: raise Exception("CRITICAL: Failed to load guidelines.")
        structured_notebook = preprocess_notebook(cells)
        status_placeholder.info(f"✅ Notebook structured with {len(cells)} cells.")

        status_placeholder.info(f"[4/7] Kicking off parallel reviews...")
        all_findings, results_queue, threads = [], Queue(), []
        for i, (guideline_name, guideline_content) in enumerate(guidelines.items()):
            model = REVIEW_MODELS[i % len(REVIEW_MODELS)]
            thread = threading.Thread(target=run_review_task, args=(results_queue, structured_notebook, model, guideline_name, guideline_content, status_placeholder))
            threads.append(thread)
            thread.start()
            time.sleep(1)
        for thread in threads: thread.join()

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
        if not all_findings:
            validation_result = {"final_report": [], "overall_feedback": "Excellent! No issues found."}
        else:
            validation_system_prompt = "You are the Senior AI Audit Validator..."
            validation_prompt = build_validation_prompt(structured_notebook, all_findings)
            response = call_gemini_api(prompt=validation_prompt, system_prompt=validation_system_prompt, model=VALIDATION_MODEL)
            if response:
                validation_result = extract_json_from_response(response) or {"final_report": [], "overall_feedback": "Validation failed: Malformed JSON from model."}
            else:
                validation_result = {"final_report": [], "overall_feedback": "Validation failed due to a Gemini API error."}
        
        status_placeholder.info("[7/7] Generating final report...")
        final_report = generate_final_report(validation_result, notebook_name)
        report_filename = f"FINAL_AUDIT_REPORT_{notebook_name.replace('.ipynb', '')}.md"
        
        status_placeholder.success("🎉 Audit Complete!")
        return final_report, report_filename, []
    
    except Exception as e:
        errors_found.append({"error": True, "guideline": "Main Workflow", "model": "N/A", "message": str(e), "traceback": traceback.format_exc()})
        return None, None, errors_found


# --- STREAMLIT USER INTERFACE (MODIFIED FOR DEBUGGING) ---
if __name__ == "__main__":
    st.set_page_config(page_title="AI Notebook Auditor [Debug Mode]", layout="wide")
    st.title("🤖 AI Notebook Auditor [Debug Mode]")
    st.markdown("Enter a task number to run the audit. This version will display detailed errors if they occur.")

    task_number = st.text_input("Enter the Task Number:", placeholder="e.g., 78123")

    if st.button("Start Review", type="primary", use_container_width=True):
        if task_number and task_number.isdigit():
            console_container = st.expander("Live Console Log", expanded=True)
            status_placeholder = console_container.empty()
            
            with st.spinner("Executing audit... waiting for thread results..."):
                final_report_md, report_filename, errors = run_audit_workflow(task_number, status_placeholder)
            
            if errors:
                st.error("One or more tasks failed! Here are the details:")
                for error in errors:
                    st.subheader(f"Error in review for: `{error['guideline']}` (using model `{error['model']}`)")
                    st.write("**Error Message:**")
                    st.write(error['message'])
                    st.write("**Full Traceback:**")
                    st.code(error['traceback'], language='text')
            elif final_report_md:
                st.balloons()
                st.header("Generated Review Content")
                st.markdown(final_report_md)
                st.download_button(
                   label="⬇️ Download Full Report", data=final_report_md,
                   file_name=report_filename, mime="text/markdown", use_container_width=True)
        else:
            st.error("❗ Please enter a valid, numeric task number.")

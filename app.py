import json
import os
import re 
import requests
import time
import docx
import threading
import traceback
import zipfile
import google.generativeai as genai
import streamlit as st
import tempfile
import firebase_admin
from firebase_admin import credentials, firestore
from queue import Queue
from urllib.parse import urlparse
from dotenv import load_dotenv
from datetime import datetime

# --- CONFIGURATION FOR MAXIMUM ACCURACY ---
load_dotenv()
REVIEW_MODEL = "gemini-2.5-pro"
VALIDATION_MODEL = "gemini-2.5-flash"
RATE_LIMIT = 5
USAGE_TRACKER_FILE = 'usage_tracker.json'
MAX_RUNS_PER_TASK = 2
usage_lock = threading.Lock()

USD_TO_INR_EXCHANGE_RATE = 85.0 
# Prices are per 1 million tokens.
GEMINI_2_5_PRO_PRICING = {
    "low_tier": {"input": 1.25, "output": 10.00, "threshold": 200000},
    "high_tier": {"input": 2.50, "output": 15.00}
}
GEMINI_2_5_FLASH_PRICING = {
    "input": 0.30, 
    "output": 2.50
}

# --- FIRESTORE INITIALIZATION ---
def initialize_firestore():
    try:
        firebase_admin.get_app()
    except ValueError:
        try:
            service_account_info = json.loads(st.secrets["FIREBASE_SERVICE_ACCOUNT"])
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Firebase initialization failed. Ensure your FIREBASE_SERVICE_ACCOUNT secret is set correctly. Error: {e}")
            return None
    return firestore.client()

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

# --- USAGE & COST TRACKER FUNCTIONS ---
def read_usage_tracker():
    with usage_lock:
        if not os.path.exists(USAGE_TRACKER_FILE):
            return {}
        try:
            with open(USAGE_TRACKER_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

def write_usage_tracker(data):
    with usage_lock:
        with open(USAGE_TRACKER_FILE, 'w') as f:
            json.dump(data, f, indent=4)

def log_audit_to_firestore(db_client, log_data):
    if db_client:
        try:
            db_client.collection('audit_logs').add(log_data)
        except Exception as e:
            st.warning(f"Could not write log to Firestore: {e}")

def calculate_cost_inr(model_name, input_tokens, output_tokens):
    """Calculates the cost in INR based on the specific model and its pricing tiers."""
    cost_usd = 0.0
    if "2.5-pro" in model_name:
        pricing = GEMINI_2_5_PRO_PRICING
        tier = "low_tier" if input_tokens <= pricing["low_tier"]["threshold"] else "high_tier"
        input_cost_usd = (input_tokens / 1_000_000) * pricing[tier]["input"]
        output_cost_usd = (output_tokens / 1_000_000) * pricing[tier]["output"]
        cost_usd = input_cost_usd + output_cost_usd
    elif "2.5-flash" in model_name:
        pricing = GEMINI_2_5_FLASH_PRICING
        input_cost_usd = (input_tokens / 1_000_000) * pricing["input"]
        output_cost_usd = (output_tokens / 1_000_000) * pricing["output"]
        cost_usd = input_cost_usd + output_cost_usd
    
    return cost_usd * USD_TO_INR_EXCHANGE_RATE

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

def call_gemini_api(prompt, system_prompt=None, model_obj=None):
    time.sleep(RATE_LIMIT)
    
    response = model_obj.generate_content(prompt, request_options={'timeout': 300})
    
    if not response.parts:
        try:
            finish_reason = response.candidates[0].finish_reason.name
            if finish_reason == "SAFETY":
                raise ValueError("The model's response was blocked by safety filters.")
            else:
                raise ValueError(f"The model stopped generating text for an unexpected reason: {finish_reason}")
        except (IndexError, AttributeError):
             raise ValueError("The model returned an empty response without a clear reason.")

    token_dict = {'input': 0, 'output': 0}
    try:
        token_dict['input'] = response.usage_metadata.prompt_token_count
        token_dict['output'] = response.usage_metadata.candidates_token_count
    except Exception:
        pass
        
    return response.text, token_dict

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
            guidelines[os.path.splitext(filename)[0].lower()] = "\n".join(para.text for para in doc.paragraphs)
    return guidelines

def extract_json_from_response(response_text):
    if not response_text: raise ValueError("API response was empty.")
    
    if response_text.strip().startswith("```json"):
        response_text = response_text.strip()[7:-3]
    elif response_text.strip().startswith("```"):
        response_text = response_text.strip()[3:-3]

    json_match = response_text[response_text.find('{'):response_text.rfind('}') + 1]

    if json_match:
        try:
            return json.loads(json_match)
        except json.JSONDecodeError as e:
            if "Expecting ',' delimiter" in str(e):
                fixed_json_match = re.sub(r'\}\s*\{', '}, {', json_match)
                try:
                    return json.loads(fixed_json_match)
                except json.JSONDecodeError:
                    raise ValueError(f"JSON auto-correction failed. Original Error: {e}. Raw Text: {response_text}")
            else:
                raise ValueError(f"JSON Parsing Failed with a non-comma error. Raw Text: {response_text}")

    raise ValueError(f"Could not find a valid JSON object in the response. Raw Text: {response_text}")


# --- PROMPT ENGINEERING & REPORTING ---
def build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content):
    return f"""Your task is to act as a specialized AI auditor operating in a "Chain of Thought" mode. For every potential violation you identify, you MUST first articulate your step-by-step reasoning in the `thinking_process` field before summarizing the issue. Your response must be a single, valid JSON object and NOTHING ELSE.

**Your Sole Focus: The "{guideline_name}" Guideline**
---
{guideline_content}
---

**Mandatory Instructions:**
1.  **Adopt a "Thinking Mode":** Before making any conclusion, you must think step-by-step.
2.  **Analyze Carefully:** Review the entire notebook for violations of ONLY the guideline specified above.
3.  **Mandatory Thinking Process:** For each violation, you MUST fill the `thinking_process` field with your detailed, step-by-step reasoning. Keep your reasoning clear but concise.
4.  **Concise Issue Description:** After your thinking process, provide a clear and concise `issue_description`.
5.  **No Violations:** If you find NO violations, you MUST return an empty `findings` array: `"findings": []`.
6.  **Strict JSON:** Your entire output must be a single, valid JSON object. Do not include any conversational text.
7.  **CRITICAL JSON SYNTAX:** If the `findings` array contains more than one object, you MUST place a comma (`,`) between each object.

**Output Examples:**

*Example 1: If you find violations*
```json
{{
  "guideline": "{guideline_name}",
  "findings": [
    {{
      "cell_number": 5,
      "thinking_process": "Step 1: I am checking for factual errors under the 'hallucination' guideline. Step 2: I see the model claims the capital of France is Berlin. Step 3: I know the correct capital is Paris. Step 4: Therefore, this is a clear factual error and a violation.",
      "issue_description": "In cell 5, the model incorrectly states that the capital of France is Berlin. The correct capital is Paris."
    }}, {{
      "cell_number": 12,
      "thinking_process": "Step 1: The model invented a function `calculate_gdp()` that does not exist in the pandas library. Step 2: This is a fabrication and a violation of the hallucination guideline.",
      "issue_description": "The model invented a non-existent function `calculate_gdp()` in cell 12."
    }}
  ]
}}
```

*Example 2: If you find NO violations*
```json
{{
  "guideline": "{guideline_name}",
  "findings": []
}}
```

**Notebook to Analyze:**
---
{structured_notebook}
---

Begin your analysis now. Remember to use your "Thinking Mode" and pay close attention to JSON syntax, especially commas. Your final output MUST BE the JSON object and nothing more.
"""

def build_validation_prompt(structured_notebook, all_findings):
    return f"""As the Senior AI Audit Validator, your task is to review all preliminary findings from a team of specialized AI auditors, validate them, and compile a final, clean report.
**Instructions:**
1.  Review all findings provided below. Each finding is marked with its 'violation_category' (e.g., 'hallucination', 'logical').
2.  Validate each finding by cross-referencing it with the notebook structure and the auditor's thinking.
3.  Eliminate duplicate findings and incorrect "false positives."
4.  For each validated issue, provide a detailed explanation and your own validation reasoning.
5.  Organize all validated issues by their cell number.
6.  Provide a concise, high-level summary of the notebook's overall quality.
7.  Return your final validated report exclusively in a valid JSON format.
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
          "violation_category": "<The name of the guideline violated, e.g., 'hallucination', 'logical', 'structure'>",
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

# --- WORKER FUNCTION (FOR INDIVIDUAL REVIEWS) ---
def run_review_task(queue, model_obj, guideline_name, guideline_content, structured_notebook, token_counts, token_lock):
    try:
        system_prompt = "You are a meticulous AI Notebook Auditor..."
        prompt = build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content)
        
        response_text, token_dict = call_gemini_api(prompt, system_prompt=system_prompt, model_obj=model_obj)
        
        with token_lock:
            token_counts.append(token_dict)

        findings_data = extract_json_from_response(response_text)
        
        if findings_data and "findings" not in findings_data:
            queue.put([])
        elif findings_data and "findings" in findings_data:
            findings = findings_data.get("findings", [])
            for finding in findings:
                finding["auditor_model"] = REVIEW_MODEL
                finding["violation_category"] = guideline_name 
            queue.put(findings)
        else:
            raise ValueError("Response was not a valid JSON object or was empty.")
            
    except Exception as e:
        queue.put([{"error": True, "guideline": guideline_name, "model": REVIEW_MODEL, "message": str(e), "traceback": traceback.format_exc()}])

# --- MAIN WORKFLOW FUNCTION ---
def run_audit_workflow(task_number, status_placeholder, db_client):
    try:
        usage_data = read_usage_tracker()
        run_count = usage_data.get(str(task_number), 0)
        if run_count >= MAX_RUNS_PER_TASK:
            raise PermissionError(f"Task {task_number} has already been reviewed the maximum number of times ({MAX_RUNS_PER_TASK}).")
        
        usage_data[str(task_number)] = run_count + 1
        write_usage_tracker(usage_data)
    
        with tempfile.TemporaryDirectory() as temp_dir:
            status_placeholder.info(f"🚀 Audit initiated for task {task_number} (Run {run_count + 1} of {MAX_RUNS_PER_TASK})...")
            AUTH_TOKEN = st.secrets["AUTH_TOKEN"]
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=GEMINI_API_KEY.strip())

            review_gen_config = genai.types.GenerationConfig(temperature=0.0, max_output_tokens=16384, response_mime_type="application/json")
            validation_gen_config = genai.types.GenerationConfig(temperature=0.0, max_output_tokens=16384)
            safety_settings = {'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE','HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE','HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE','HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'}
            
            review_model_obj = genai.GenerativeModel(REVIEW_MODEL, generation_config=review_gen_config, safety_settings=safety_settings)
            validation_model_obj = genai.GenerativeModel(VALIDATION_MODEL, generation_config=validation_gen_config, safety_settings=safety_settings)

            status_placeholder.info(f"[1/5] Downloading & Preprocessing...")
            notebook_zip_url = f"https://labeling-s.turing.com/api/conversations/download-notebook-zip?ids[]={task_number}"
            downloader = DataDownloader(AUTH_TOKEN)
            zip_path = downloader.download_zip_file(notebook_zip_url, temp_dir)
            notebook_path = find_and_unzip(zip_path, temp_dir)
            notebook_name = os.path.basename(notebook_path)
            cells = load_notebook_cells(notebook_path)
            structured_notebook = preprocess_notebook(cells)
            all_guidelines = load_guidelines("./Guidedalines")
            status_placeholder.info("✅ Download & Preprocessing Complete.")

            status_placeholder.info("[2/5] Kicking off individual parallel reviews...")
            all_findings, results_queue = [], Queue()
            threads, token_counts, token_lock = [], [], threading.Lock()
            for guideline_name, guideline_content in all_guidelines.items():
                status_placeholder.info(f"   - Dispatching '{guideline_name}' review to {REVIEW_MODEL}...")
                thread = threading.Thread(target=run_review_task, args=(results_queue, review_model_obj, guideline_name, guideline_content, structured_notebook, token_counts, token_lock))
                threads.append(thread); thread.start()
                time.sleep(1)
            
            for thread in threads: 
                thread.join()
            status_placeholder.info("✅ All review threads finished. Aggregating results...")

            errors_found = []
            while not results_queue.empty():
                result_list = results_queue.get()
                for result in result_list:
                    if isinstance(result, dict) and result.get("error"):
                        errors_found.append(result)
                    else:
                        all_findings.append(result)
            
            if errors_found:
                return None, None, errors_found, {}, []

            status_placeholder.info(f"[3/5] Aggregated {len(all_findings)} findings.")
            status_placeholder.info(f"[4/5] Running final validation with {VALIDATION_MODEL}...")
            
            validation_token_dict = {'input': 0, 'output': 0}
            if not all_findings:
                validation_result = {"final_report": [], "overall_feedback": "Excellent! No issues were found by the specialized auditors."}
            else:
                validation_prompt = build_validation_prompt(structured_notebook, all_findings)
                response_text, validation_token_dict = call_gemini_api(prompt=validation_prompt, system_prompt="You are the Senior AI Audit Validator...", model_obj=validation_model_obj)
                validation_result = extract_json_from_response(response_text) if response_text else None
                if not validation_result:
                     validation_result = {"final_report": [], "overall_feedback": "Validation step failed."}
            
            total_input_tokens = 0
            total_output_tokens = 0
            total_cost_inr = 0

            for tc in token_counts:
                total_input_tokens += tc['input']
                total_output_tokens += tc['output']
                total_cost_inr += calculate_cost_inr(REVIEW_MODEL, tc['input'], tc['output'])

            total_input_tokens += validation_token_dict['input']
            total_output_tokens += validation_token_dict['output']
            total_cost_inr += calculate_cost_inr(VALIDATION_MODEL, validation_token_dict['input'], validation_token_dict['output'])
            
            grand_total_tokens = total_input_tokens + total_output_tokens

            final_token_summary = {
                'input': total_input_tokens,
                'output': total_output_tokens,
                'total': grand_total_tokens
            }

            status_placeholder.info("[5/5] Generating final report...")
            final_report = generate_final_report(validation_result, notebook_name)
            report_filename = f"FINAL_AUDIT_REPORT_{notebook_name.replace('.ipynb', '')}.md"
            
            log_entry = {
                "timestamp": datetime.now(),
                "task_number": task_number,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": grand_total_tokens,
                "estimated_cost_inr": total_cost_inr
            }
            log_audit_to_firestore(db_client, log_entry)
            
            status_placeholder.success("🎉 Audit Complete!")
            return final_report, report_filename, [], final_token_summary, []
        
    except Exception as e:
        return None, None, [{"error": True, "guideline": "Pre-flight Check", "model": "N/A", "message": str(e), "traceback": traceback.format_exc()}], {}, []

# --- STREAMLIT UI ---
if __name__ == "__main__":
    st.set_page_config(page_title="AI Notebook Auditor", layout="wide")
    st.title("🤖 AI Notebook Auditor")
    st.markdown("Enter a task number to perform a multi-model review. Each task can be reviewed a maximum of two times.")
    
    db = initialize_firestore()

    task_number = st.text_input("Enter the Task Number:", placeholder="e.g., 214514")

    if st.button("Start Review", type="primary", use_container_width=True):
        if db is None:
            st.error("Firestore database is not connected. Please check your service account credentials.")
        elif task_number and task_number.isdigit():
            console_container = st.expander("Live Console Log", expanded=True)
            status_placeholder = console_container.empty()
            with st.spinner("Executing audit..."):
                final_report_md, report_filename, errors, token_summary, warnings = run_audit_workflow(task_number, status_placeholder, db)
            
            if warnings:
                for warning_msg in set(warnings):
                    st.warning(f"⚠️ **Notice:** {warning_msg}")

            if errors:
                st.error("The audit could not be completed:")
                for error in errors:
                    if "PermissionError" in error.get('traceback', ''):
                         st.warning(error.get('message'))
                    else:
                        if error.get('guideline') != 'Pre-flight Check':
                            st.subheader(f"Error during: `{error.get('guideline', 'Unknown Step')}` review (Model: `{error.get('model', 'N/A')}`)")
                        st.write("**Message:**")
                        st.code(error.get('message', 'No message.'), language='text')
                        st.write("**Full Traceback:**")
                        st.code(error.get('traceback', 'No traceback.'), language='text')

            elif final_report_md:
                st.info(f"📊 **Token Usage:** Input: {token_summary.get('input', 0):,} | Output: {token_summary.get('output', 0):,} | **Total: {token_summary.get('total', 0):,}**")
                st.balloons()
                st.header("Generated Review Content")
                st.markdown(final_report_md)
                st.download_button(label="⬇️ Download Full Report", data=final_report_md, file_name=report_filename, use_container_width=True)
        else:
            st.error("❗ Please enter a valid, numeric task number.")

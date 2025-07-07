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
REVIEW_MODEL = "gemini-1.5-pro"
VALIDATION_MODEL = "gemini-1.5-flash"
RATE_LIMIT = 5  # Seconds to wait between API calls
USAGE_TRACKER_FILE = 'usage_tracker.json'
MAX_RUNS_PER_TASK = 2
usage_lock = threading.Lock()

USD_TO_INR_EXCHANGE_RATE = 85.0
# Prices are per 1 million tokens.
GEMINI_1_5_PRO_PRICING = {
    "low_tier": {"input": 3.50, "output": 10.50, "threshold": 128000},
    "high_tier": {"input": 7.00, "output": 21.00}
}
GEMINI_1_5_FLASH_PRICING = {
    "input": 0.35,
    "output": 1.05
}


# --- FIRESTORE INITIALIZATION ---
def initialize_firestore():
    """Initializes and returns a Firestore client, handling potential errors."""
    try:
        # Check if the app is already initialized
        firebase_admin.get_app()
    except ValueError:
        try:
            # Load credentials from Streamlit secrets for deployment
            service_account_info_str = st.secrets.get("FIREBASE_SERVICE_ACCOUNT")
            if not service_account_info_str:
                st.error("Firebase service account secret is not set.")
                return None
            service_account_info = json.loads(service_account_info_str)
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(
                f"Firebase initialization failed. Ensure your FIREBASE_SERVICE_ACCOUNT secret is set correctly. Error: {e}")
            return None
    return firestore.client()


class DataDownloader:
    """Handles downloading files with authorization."""
    def __init__(self, auth_token):
        self.auth_token = auth_token

    def download_zip_file(self, url, save_path):
        """Downloads a ZIP file from a URL using a bearer token."""
        try:
            clean_auth_token = self.auth_token.strip()
            headers = {"Authorization": f"Bearer {clean_auth_token}"}
            clean_url = url.strip()
            # Set a generous timeout for the download request
            response = requests.get(clean_url, headers=headers, stream=True, timeout=120)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # Extract task ID from URL to create a unique filename
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
    """Reads the usage tracker file safely with a lock."""
    with usage_lock:
        if not os.path.exists(USAGE_TRACKER_FILE):
            return {}
        try:
            with open(USAGE_TRACKER_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}


def write_usage_tracker(data):
    """Writes to the usage tracker file safely with a lock."""
    with usage_lock:
        with open(USAGE_TRACKER_FILE, 'w') as f:
            json.dump(data, f, indent=4)


def log_audit_to_firestore(db_client, log_data):
    """Logs audit metadata to a Firestore collection."""
    if db_client:
        try:
            db_client.collection('audit_logs').add(log_data)
        except Exception as e:
            st.warning(f"Could not write log to Firestore: {e}")


def calculate_cost_inr(model_name, input_tokens, output_tokens):
    """Calculates the cost in INR based on the specific model and its pricing tiers."""
    cost_usd = 0.0
    if "1.5-pro" in model_name:
        pricing = GEMINI_1_5_PRO_PRICING
        # Determine pricing tier based on input token count
        tier = "low_tier" if input_tokens <= pricing["low_tier"]["threshold"] else "high_tier"
        input_cost_usd = (input_tokens / 1_000_000) * pricing[tier]["input"]
        output_cost_usd = (output_tokens / 1_000_000) * pricing[tier]["output"]
        cost_usd = input_cost_usd + output_cost_usd
    elif "1.5-flash" in model_name:
        pricing = GEMINI_1_5_FLASH_PRICING
        input_cost_usd = (input_tokens / 1_000_000) * pricing["input"]
        output_cost_usd = (output_tokens / 1_000_000) * pricing["output"]
        cost_usd = input_cost_usd + output_cost_usd

    return cost_usd * USD_TO_INR_EXCHANGE_RATE


# --- UTILITY AND LOGIC FUNCTIONS ---
def find_and_unzip(zip_path, extract_folder):
    """Unzips a file and finds the first .ipynb file within it."""
    os.makedirs(extract_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    # Walk through the extracted files to find the notebook
    for root, _, files in os.walk(extract_folder):
        for file in files:
            if file.endswith(".ipynb"):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No .ipynb file found in the extracted content of {zip_path}")


def call_gemini_api(prompt, model_obj):
    """Calls the Gemini API and handles potential empty or blocked responses."""
    time.sleep(RATE_LIMIT)  # Respect rate limits

    # Generate content with a timeout
    response = model_obj.generate_content(prompt, request_options={'timeout': 300})

    # Handle cases where the response is blocked or empty
    if not response.parts:
        try:
            finish_reason = response.candidates[0].finish_reason.name
            if finish_reason == "SAFETY":
                raise ValueError("The model's response was blocked by safety filters.")
            else:
                raise ValueError(f"The model stopped generating text for an unexpected reason: {finish_reason}")
        except (IndexError, AttributeError):
            raise ValueError("The model returned an empty response without a clear reason.")

    # Extract token usage metadata
    token_dict = {'input': 0, 'output': 0}
    try:
        token_dict['input'] = response.usage_metadata.prompt_token_count
        token_dict['output'] = response.usage_metadata.candidates_token_count
    except Exception:
        # If token count isn't available, we'll just use 0
        pass

    return response.text, token_dict


def load_notebook_cells(file_path):
    """Loads the 'cells' from a Jupyter notebook file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f).get('cells', [])


def preprocess_notebook(cells):
    """
    Processes notebook cells to create a structured JSON for the AI and a role map for the final report.
    Returns:
        - A JSON string representing the structured notebook for AI analysis.
        - A dictionary mapping each cell number to its identified role (e.g., 'assistant_response').
    """
    structured, turn_counter = {}, 0
    cell_role_map = {}  # Map to store role for each cell number
    for i, cell in enumerate(cells):
        source, cell_num, role = "".join(cell.get('source', [])), i + 1, "unknown"
        # Markers to identify the role of each cell
        markers = {
            "**[system]**": "system_prompt",
            "**[user]**": "user_query",
            "**[assistant]**": "assistant_response",
            "**[thinking]**": "assistant_thinking",
            "**[thought]**": "assistant_thought",
            "**[tool_use]**": "tool_use",
            "**[tool_output]**": "tool_output",
            "**[tools]**": "tool_definitions"
        }
        for marker, r in markers.items():
            if marker in source:
                role = r
                break

        cell_role_map[cell_num] = role  # Store the role for the final report

        # Group cells by user turns
        if role == "user_query":
            turn_counter += 1
            structured[f"turn_{turn_counter}"] = []
        if role != "unknown":
            if f"turn_{turn_counter}" not in structured:
                if turn_counter == 0: turn_counter = 1
                structured[f"turn_{turn_counter}"] = []
            structured[f"turn_{turn_counter}"].append({"cell_number": cell_num, "role": role, "content": source})

    return json.dumps(structured, indent=2), cell_role_map


def load_guidelines(guidelines_dir):
    """Loads all .docx guideline files from a directory."""
    guidelines = {}
    for filename in sorted(os.listdir(guidelines_dir)):
        # Ignore temporary Word files
        if filename.endswith(".docx") and not filename.startswith("~$"):
            file_path = os.path.join(guidelines_dir, filename)
            doc = docx.Document(file_path)
            guidelines[os.path.splitext(filename)[0].lower()] = "\n".join(para.text for para in doc.paragraphs)
    return guidelines


def extract_json_from_response(response_text):
    """
    Extracts a JSON object from a model's text response, attempting to clean and fix common errors.
    """
    if not response_text:
        raise ValueError("API response was empty.")

    # Clean markdown code fences
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:].strip()
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3].strip()
    
    # Find the outermost JSON object or array
    json_match = ""
    start_obj, end_obj = cleaned_text.find('{'), cleaned_text.rfind('}')
    start_arr, end_arr = cleaned_text.find('['), cleaned_text.rfind(']')

    if start_obj != -1 and end_obj != -1:
        json_match = cleaned_text[start_obj : end_obj + 1]
    elif start_arr != -1 and end_arr != -1:
        json_match = cleaned_text[start_arr : end_arr + 1]

    if not json_match:
        raise ValueError(f"Could not find a valid JSON object or array in the response. Raw Text: {response_text}")

    try:
        return json.loads(json_match)
    except json.JSONDecodeError as e:
        # Attempt to fix common JSON errors like missing commas
        # This regex inserts a comma between closing and opening curly braces
        fixed_json_match = re.sub(r'(?<=\})\s*(?=\{)', ',', json_match)
        # This regex removes trailing commas before a closing bracket or brace
        fixed_json_match = re.sub(r',\s*([\]}])', r'\1', fixed_json_match)
        try:
            return json.loads(fixed_json_match)
        except json.JSONDecodeError as e_fixed:
            raise ValueError(f"JSON auto-correction failed. Original Error: {e}. Fixed Error: {e_fixed}. Raw Text: {response_text}")


# --- PROMPT ENGINEERING & REPORTING ---
def build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content):
    """Builds a prompt for a specialized auditor AI to review a notebook against one guideline."""
    return f"""Your task is to act as a specialized AI auditor. Your response must be a single, valid JSON object and NOTHING ELSE.

**Your Sole Focus: The "{guideline_name}" Guideline**
---
{guideline_content}
---

**Mandatory Instructions:**
1.  **Chain of Thought:** For each violation, you MUST first articulate your step-by-step reasoning in the `thinking_process` field.
2.  **Analyze Carefully:** Review the entire notebook for violations of ONLY the guideline specified above.
3.  **Concise Issue Description:** After your thinking process, provide a clear and concise `issue_description`.
4.  **No Violations:** If you find NO violations, you MUST return an empty `findings` array: `"findings": []`.
5.  **Strict JSON:** Your entire output must be a single, valid JSON object. Do not include any conversational text.
6.  **CRITICAL JSON SYNTAX:** If the `findings` array contains more than one object, you MUST place a comma (`,`) between each object.

**Output Example (if violations are found):**
```json
{{
  "guideline": "{guideline_name}",
  "findings": [
    {{
      "cell_number": 5,
      "thinking_process": "Step 1: I am checking for factual errors under the 'hallucination' guideline. Step 2: I see the model claims the capital of France is Berlin. Step 3: I know the correct capital is Paris. Step 4: Therefore, this is a clear factual error and a violation.",
      "issue_description": "In cell 5, the model incorrectly states that the capital of France is Berlin."
    }}
  ]
}}
```

**Notebook to Analyze:**
---
{structured_notebook}
---

Begin your analysis. Your final output MUST BE the JSON object.
"""


def build_validation_prompt(structured_notebook, all_findings):
    """Builds a prompt for a senior validator AI to review all findings and create a final report."""
    return f"""As the Senior AI Audit Validator, your task is to review all preliminary findings, validate them, and compile a final, clean report in JSON format.
**Instructions:**
1.  Review all findings provided below.
2.  Validate each finding by cross-referencing it with the notebook.
3.  Eliminate duplicate findings and "false positives."
4.  Organize all validated issues by their cell number.
5.  Provide a concise, high-level summary of the notebook's overall quality.
6.  Return your final validated report exclusively in a valid JSON format.

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
          "violation_category": "<The name of the guideline violated, e.g., 'hallucination'>",
          "problematic_content": "<Quote the exact problematic text or summarize the action.>",
          "reason_for_violation": "<Explain why this is a violation of the guidelines.>",
          "validation_reasoning": "<Explain your thought process for validating this issue.>"
        }}
      ]
    }}
  ],
  "overall_feedback": "<A brief, high-level summary of the audit results.>"
}}
```"""


def generate_final_report(validation_result, notebook_name, cell_role_map):
    """Generates a markdown report from the validation results, including the cell type."""
    report = f"# Final Audit Report: {notebook_name}\n"
    report += f"**Generated at**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
    overall_feedback = validation_result.get("overall_feedback", "No overall feedback provided.")
    report += f"## Overall Feedback\n{overall_feedback}\n\n---\n\n"
    report += "## Cell-wise Findings\n\n"
    final_report_data = validation_result.get("final_report", [])
    if not final_report_data:
        report += "✅ No validated issues found in the final report.\n"
    else:
        # A map to format the role names for the report header
        display_role_map = {
            "system_prompt": "[system]", "user_query": "[user]", "assistant_response": "[assistant]",
            "assistant_thinking": "[thinking]", "assistant_thought": "[thought]", "tool_use": "[tool_use]",
            "tool_output": "[tool_output]", "tool_definitions": "[tools]", "unknown": "[unknown]"
        }
        sorted_report_data = sorted(final_report_data, key=lambda x: x.get("cell_number", float('inf')))
        for cell_data in sorted_report_data:
            cell_num = cell_data.get("cell_number")
            issues = cell_data.get("validated_issues", [])

            if issues and cell_num is not None:
                # Look up the role from the map created during preprocessing
                role = cell_role_map.get(cell_num, 'unknown')
                display_role = display_role_map.get(role, f"[{role}]")
                report += f"### Cell {cell_num} {display_role.strip()}\n\n"

                for issue in issues:
                    # FIX: Corrected typo from 'viulation_category' to 'violation_category'
                    report += f"**Violation Category:** {issue.get('violation_category', 'N/A')}\n\n"
                    report += f"**Problematic Content:**\n```\n{issue.get('problematic_content', 'N/A')}\n```\n\n"
                    report += f"**Reason for Violation:**\n{issue.get('reason_for_violation', 'N/A')}\n\n"
                    report += f"**Validator's Reasoning:**\n*_{issue.get('validation_reasoning', 'N/A')}_*\n\n"
                report += "---\n"
    report += "\n*Report generated with a multi-model audit via Google Gemini.*"
    return report


# --- WORKER FUNCTION (FOR INDIVIDUAL REVIEWS) ---
def run_review_task(queue, model_obj, guideline_name, guideline_content, structured_notebook, token_counts, token_lock):
    """The function executed by each thread to review one guideline."""
    try:
        prompt = build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content)
        response_text, token_dict = call_gemini_api(prompt, model_obj=model_obj)

        # Safely update shared token counts
        with token_lock:
            token_counts.append(token_dict)

        findings_data = extract_json_from_response(response_text)
        findings = findings_data.get("findings", [])

        # Add metadata to each finding
        for finding in findings:
            finding["auditor_model"] = REVIEW_MODEL
            finding["violation_category"] = guideline_name
        queue.put(findings)

    except Exception as e:
        # Put an error object in the queue for the main thread to handle
        queue.put([{"error": True, "guideline": guideline_name, "model": REVIEW_MODEL, "message": str(e),
                    "traceback": traceback.format_exc()}])


# --- MAIN WORKFLOW FUNCTION ---
def run_audit_workflow(task_number, status_placeholder, db_client):
    """Orchestrates the entire audit process from download to final report."""
    try:
        # --- PRE-FLIGHT CHECKS ---
        usage_data = read_usage_tracker()
        run_count = usage_data.get(str(task_number), 0)
        if run_count >= MAX_RUNS_PER_TASK:
            raise PermissionError(
                f"Task {task_number} has already been reviewed the maximum number of times ({MAX_RUNS_PER_TASK}).")

        usage_data[str(task_number)] = run_count + 1
        write_usage_tracker(usage_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            status_placeholder.info(
                f"🚀 Audit initiated for task {task_number} (Run {run_count + 1} of {MAX_RUNS_PER_TASK})...")
            
            # --- INITIALIZATION ---
            AUTH_TOKEN = st.secrets["AUTH_TOKEN"]
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=GEMINI_API_KEY.strip())

            review_gen_config = genai.types.GenerationConfig(temperature=0.0, response_mime_type="application/json")
            validation_gen_config = genai.types.GenerationConfig(temperature=0.0)
            safety_settings = {'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE', 'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                               'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                               'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'}

            review_model_obj = genai.GenerativeModel(REVIEW_MODEL, generation_config=review_gen_config, safety_settings=safety_settings)
            validation_model_obj = genai.GenerativeModel(VALIDATION_MODEL, generation_config=validation_gen_config, safety_settings=safety_settings)

            # --- STEP 1: DOWNLOAD & PREPROCESS ---
            status_placeholder.info(f"[1/5] Downloading & Preprocessing...")
            notebook_zip_url = f"https://labeling-s.turing.com/api/conversations/download-notebook-zip?ids[]={task_number}"
            downloader = DataDownloader(AUTH_TOKEN)
            zip_path = downloader.download_zip_file(notebook_zip_url, temp_dir)
            notebook_path = find_and_unzip(zip_path, temp_dir)
            notebook_name = os.path.basename(notebook_path)
            cells = load_notebook_cells(notebook_path)
            structured_notebook, cell_role_map = preprocess_notebook(cells)
            all_guidelines = load_guidelines("./Guidelines")
            status_placeholder.info("✅ Download & Preprocessing Complete.")

            # --- STEP 2: PARALLEL REVIEWS ---
            status_placeholder.info("[2/5] Kicking off parallel reviews...")
            all_findings, results_queue = [], Queue()
            threads, token_counts, token_lock = [], [], threading.Lock()
            for guideline_name, guideline_content in all_guidelines.items():
                status_placeholder.info(f"  - Dispatching '{guideline_name}' review to {REVIEW_MODEL}...")
                thread = threading.Thread(target=run_review_task, args=(
                    results_queue, review_model_obj, guideline_name, guideline_content, structured_notebook, token_counts,
                    token_lock))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
            status_placeholder.info("✅ All review threads finished. Aggregating results...")

            # --- STEP 3: AGGREGATE & VALIDATE ---
            errors_found = []
            while not results_queue.empty():
                result_list = results_queue.get()
                # FIX: The loop should iterate over 'result_list', not 'result'
                for result_item in result_list:
                    if isinstance(result_item, dict) and result_item.get("error"):
                        errors_found.append(result_item)
                    else:
                        all_findings.append(result_item)

            if errors_found:
                return None, None, errors_found, {}, 0

            status_placeholder.info(f"[3/5] Aggregated {len(all_findings)} potential findings.")
            status_placeholder.info(f"[4/5] Running final validation with {VALIDATION_MODEL}...")

            validation_token_dict = {'input': 0, 'output': 0}
            if not all_findings:
                validation_result = {"final_report": [],
                                     "overall_feedback": "Excellent! No issues were found by the specialized auditors."}
            else:
                validation_prompt = build_validation_prompt(structured_notebook, all_findings)
                response_text, validation_token_dict = call_gemini_api(prompt=validation_prompt, model_obj=validation_model_obj)
                validation_result = extract_json_from_response(response_text) if response_text else None
                if not validation_result:
                    validation_result = {"final_report": [], "overall_feedback": "Validation step failed to produce a valid JSON report."}

            # --- STEP 4: CALCULATE COSTS ---
            total_input_tokens, total_output_tokens, total_cost_inr = 0, 0, 0.0
            for tc in token_counts:
                total_input_tokens += tc.get('input', 0)
                total_output_tokens += tc.get('output', 0)
                total_cost_inr += calculate_cost_inr(REVIEW_MODEL, tc.get('input', 0), tc.get('output', 0))

            total_input_tokens += validation_token_dict.get('input', 0)
            total_output_tokens += validation_token_dict.get('output', 0)
            total_cost_inr += calculate_cost_inr(VALIDATION_MODEL, validation_token_dict.get('input', 0), validation_token_dict.get('output', 0))

            final_token_summary = {
                'input': total_input_tokens,
                'output': total_output_tokens,
                'total': total_input_tokens + total_output_tokens
            }

            # --- STEP 5: GENERATE FINAL REPORT & LOG ---
            status_placeholder.info("[5/5] Generating final report...")
            final_report = generate_final_report(validation_result, notebook_name, cell_role_map)
            report_filename = f"FINAL_AUDIT_REPORT_{notebook_name.replace('.ipynb', '')}.md"

            log_entry = {
                "timestamp": datetime.now(), "task_number": task_number,
                "input_tokens": total_input_tokens, "output_tokens": total_output_tokens,
                "total_tokens": final_token_summary['total'], "estimated_cost_inr": total_cost_inr
            }
            log_audit_to_firestore(db_client, log_entry)

            status_placeholder.success("🎉 Audit Complete!")
            return final_report, report_filename, [], final_token_summary, total_cost_inr

    except Exception as e:
        # Catch-all for any other exceptions during the workflow
        return None, None, [{"error": True, "guideline": "Workflow Error", "model": "N/A", "message": str(e),
                             "traceback": traceback.format_exc()}], {}, 0


# --- STREAMLIT UI ---
if __name__ == "__main__":
    st.set_page_config(page_title="AI Notebook Auditor", layout="wide")
    st.title("🤖 AI Notebook Auditor")
    st.markdown(
        f"Enter a task number to perform a multi-model review. Each task can be reviewed a maximum of **{MAX_RUNS_PER_TASK}** times.")

    db = initialize_firestore()

    task_number = st.text_input("Enter the Task Number:", placeholder="e.g., 214514")

    if st.button("Start Review", type="primary", use_container_width=True):
        if db is None:
            st.error("Firestore database is not connected. Please check your service account credentials.")
        elif task_number and task_number.isdigit():
            console_container = st.expander("Live Console Log", expanded=True)
            status_placeholder = console_container.empty()
            with st.spinner("Executing multi-model audit... This may take a few minutes."):
                final_report_md, report_filename, errors, token_summary, total_cost_inr = run_audit_workflow(
                    task_number, status_placeholder, db)

            if errors:
                st.error("The audit could not be completed due to the following errors:")
                for error in errors:
                    # Special handling for permission errors which are not code failures
                    if "PermissionError" in error.get('traceback', ''):
                        st.warning(error.get('message'))
                    else:
                        st.subheader(
                            f"Error during: `{error.get('guideline', 'Unknown Step')}` review (Model: `{error.get('model', 'N/A')}`)")
                        st.write("**Message:**")
                        st.code(error.get('message', 'No message.'), language='text')
                        st.write("**Full Traceback:**")
                        st.code(error.get('traceback', 'No traceback.'), language='text')

            elif final_report_md:
                st.success("Audit complete!")
                st.info(
                    f"📊 **Token Usage:** Input: {token_summary.get('input', 0):,} | Output: {token_summary.get('output', 0):,} | **Total: {token_summary.get('total', 0):,}**\n\n"
                    f"💰 **Estimated Cost:** ₹{total_cost_inr:.2f}"
                )
                st.balloons()
                st.header("Generated Audit Report")
                st.markdown(final_report_md)
                st.download_button(label="⬇️ Download Full Report", data=final_report_md, file_name=report_filename,
                                  use_container_width=True)
        else:
            st.error("❗ Please enter a valid, numeric task number.")

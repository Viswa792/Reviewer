# app.py (Golden Master Version - Using User-Specified Original Models)
# Final version as of Monday, July 8, 2024.
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

# --- CONFIGURATION & MODELS (REVERTED TO YOUR ORIGINAL SPECIFICATION) ---
load_dotenv()
REVIEW_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]
VALIDATION_MODEL = "gemini-2.5-pro"
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

# --- API CALL FUNCTION (RESTORED TO YOUR ORIGINAL LOGIC) ---
def call_gemini_api(prompt, system_prompt=None, model=VALIDATION_MODEL):
    """
    Calls the Google Gemini API, using JSON Mode CONDITIONALLY for supported models.
    """
    time.sleep(RATE_LIMIT)
    safety_settings = {
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
    }

    # Conditional JSON Mode, as in your original script
    if "flash" in model:
        generation_config = genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=8192
        )
    else:
        generation_config = genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=8192,
            response_mime_type="application/json"
        )

    model_obj = genai.GenerativeModel(
        model_name=model,
        safety_settings=safety_settings,
        generation_config=generation_config,
        system_instruction=system_prompt
    )
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
    raise ValueError(f"JSON Parsing Failed. The model may have returned conversational text instead of JSON. Raw Text: {response_text}")

# --- PROMPT ENGINEERING & REPORTING ---
def build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content):
    # This is the new, heavily revised prompt with strong instructions and few-shot examples.
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

Begin your analysis now. Your final output MUST BE the JSON object and nothing more.
"""

def build_validation_prompt(structured_notebook, all_findings):
    # This is your original, unchanged validation prompt.
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
    # This is your original, unchanged reporting function.
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

# --- WORKER FUNCTION (SILENT, RESILIENT, AND WITH ERROR CATCHING) ---
def run_review_task(queue, structured_notebook, model_name, guideline_name, guideline_content):
    try:
        system_prompt = "You are a meticulous AI Notebook Auditor. Your task is to review a Jupyter Notebook against a specific guideline and return findings in a strict JSON format, following the examples provided in the user prompt."
        prompt = build_targeted_review_prompt(structured_notebook, guideline_name, guideline_content)
        response = call_gemini_api(prompt, system_prompt=system_prompt, model=model_name)
        findings_data = extract_json_from_response(response)
        
        if findings_data and "findings" not in findings_data:
            queue.put([]) # Success with 0 findings
        elif findings_data and "findings" in findings_data:
            findings = findings_data.get("findings", [])
            for finding in findings:
                finding["auditor"] = model_name
                finding["guideline"] = guideline_name
            queue.put(findings)
        else:
            raise ValueError("Response was not a valid JSON object or was empty.")
            
    except Exception as e:
        error_details = {
            "error": True,
            "guideline": guideline_name,
            "model": model_name,
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        queue.put(error_details)

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
        status_placeholder.info(f"[6/7] Running validation with {VALIDATION_MODEL}...")
        if not all_findings:
            validation_result = {"final_report": [], "overall_feedback": "Excellent! No issues were found by the specialized auditors."}
        else:
            validation_prompt = build_validation_prompt(structured_notebook, all_findings)
            response = call_gemini_api(prompt=validation_prompt, system_prompt="You are the Senior AI Audit Validator...", model=VALIDATION_MODEL)
            validation_result = extract_json_from_response(response) if response else None
            if not validation_result:
                 validation_result = {"final_report": [], "overall_feedback": "Validation step failed: Could not get a valid JSON response from the validation model."}
        
        status_placeholder.info("[7/7] Generating final report...")
        final_report = generate_final_report(validation_result, notebook_name)
        report_filename = f"FINAL_AUDIT_REPORT_{notebook_name.replace('.ipynb', '')}.md"
        
        status_placeholder.success("🎉 Audit Complete!")
        return final_report, report_filename, []
    
    except Exception as e:
        return None, None, [{"error": True, "guideline": "Main Workflow", "model": "N/A", "message": str(e), "traceback": traceback.format_exc()}]

# --- STREAMLIT UI ---
if __name__ == "__main__":
    st.set_page_config(page_title="AI Notebook Auditor", layout="wide")
    st.title("🤖 AI Notebook Auditor")
    st.markdown("Enter a task number to perform a multi-model review.")
    
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
                st.download_button(label="⬇️ Download Full Report", data=final_report_md, file_name=report_filename, use_container_width=True)
        else:
            st.error("❗ Please enter a valid, numeric task number.")


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
import tempfile
from queue import Queue
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()
# The models are now defined by the batch they belong to.
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

def call_gemini_api(prompt, system_prompt=None, model=VALIDATION_MODEL):
    time.sleep(RATE_LIMIT)
    safety_settings = {'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE','HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE','HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE','HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'}
    
    # Conditional JSON Mode for non-Flash models
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
            guidelines[os.path.splitext(filename)[0].lower()] = "\n".join(para.text for para in doc.paragraphs)
    return guidelines

def extract_json_from_response(response_text):
    if not response_text: raise ValueError("API response was empty.")
    if response_text.strip().startswith("```json"): response_text = response_text.strip()[7:-3]
    elif response_text.strip().startswith("```"): response_text = response_text.strip()[3:-3]
    json_match = response_text[response_text.find('{'):response_text.rfind('}') + 1]
    if json_match: return json.loads(json_match)
    raise ValueError(f"JSON Parsing Failed. Raw Text: {response_text}")

# --- PROMPT ENGINEERING & REPORTING ---
def build_batched_review_prompt(structured_notebook, guideline_batch):
    guideline_names = list(guideline_batch.keys())
    
    combined_guideline_content = ""
    for name, content in guideline_batch.items():
        combined_guideline_content += f"--- Guideline: {name} ---\n{content}\n\n"

    return f"""Your task is to act as a specialized AI auditor. You will analyze a Jupyter Notebook against a batch of guidelines simultaneously.
Your response MUST be a single, valid JSON object and NOTHING ELSE.

**Your Focus: The following guidelines: {', '.join(guideline_names)}**
{combined_guideline_content}

**Mandatory Instructions:**
1.  Analyze the entire notebook provided below.
2.  For each violation you find, you MUST identify which guideline was violated.
3.  **Crucially**: Your JSON response MUST include a `violation_category` key for each finding, specifying the name of the guideline that was violated (e.g., "hallucination", "logical").
4.  If you find NO violations for ANY of the provided guidelines, you MUST return an empty `findings` array.
5.  Your response MUST conform to the JSON schema specified in the example.

**Output Example:**
```json
{{
  "findings": [
    {{
      "violation_category": "hallucination",
      "cell_number": 5,
      "thinking_process": "The user asked for the capital of France. The model responded with \\"The capital of France is Berlin.\\". This is a factual error under the hallucination guideline.",
      "issue_description": "In cell 5, the model incorrectly states that the capital of France is Berlin."
    }},
    {{
      "violation_category": "structure",
      "cell_number": 8,
      "thinking_process": "The code in cell 8 is poorly formatted and lacks comments, making it hard to understand. This violates the structure guideline.",
      "issue_description": "The code in cell 8 should be formatted according to PEP 8 standards and include comments explaining its purpose."
    }}
  ]
}}
```

**Notebook to Analyze:**
---
{structured_notebook}
---

Begin your analysis now. Your final output MUST BE the JSON object and nothing more.
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

# --- WORKER FUNCTION (ADAPTED FOR BATCHES) ---
def run_batched_review_task(queue, structured_notebook, model_name, guideline_batch):
    try:
        guideline_names = list(guideline_batch.keys())
        system_prompt = "You are a meticulous AI Notebook Auditor. You will review a Jupyter Notebook against a batch of guidelines and return findings in a strict JSON format, tagging each finding with its violation category."
        
        prompt = build_batched_review_prompt(structured_notebook, guideline_batch)
        response = call_gemini_api(prompt, system_prompt=system_prompt, model=model_name)
        
        findings_data = extract_json_from_response(response)
        
        if findings_data and "findings" in findings_data:
            findings = findings_data.get("findings", [])
            for finding in findings:
                finding["auditor_model"] = model_name # Tag the finding with the model used
            queue.put(findings)
        else:
            queue.put([]) # Assume no findings if key is missing or data is malformed
            
    except Exception as e:
        queue.put([{"error": True, "guideline": f"Batch: {', '.join(guideline_names)}", "model": model_name, "message": str(e), "traceback": traceback.format_exc()}])

# --- MAIN WORKFLOW FUNCTION (MODIFIED FOR COST OPTIMIZATION) ---
def run_audit_workflow(task_number, status_placeholder):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            status_placeholder.info("🚀 Audit initiated...")
            AUTH_TOKEN = st.secrets["AUTH_TOKEN"]
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=GEMINI_API_KEY.strip())

            status_placeholder.info(f"[1/5] Downloading & Preprocessing...")
            notebook_zip_url = f"https://labeling-s.turing.com/api/conversations/download-notebook-zip?ids[]={task_number}"
            downloader = DataDownloader(AUTH_TOKEN)
            zip_path = downloader.download_zip_file(notebook_zip_url, temp_dir)
            notebook_path = find_and_unzip(zip_path, temp_dir)
            notebook_name = os.path.basename(notebook_path)
            cells = load_notebook_cells(notebook_path)
            structured_notebook = preprocess_notebook(cells)
            all_guidelines = load_guidelines("./Guidlines")
            status_placeholder.info("✅ Download & Preprocessing Complete.")

            status_placeholder.info("[2/5] Setting up review batches...")
            batches = [
                {
                    "model": "gemini-2.5-pro",
                    "guidelines": {
                        "hallucination": all_guidelines.get("hallucination", ""),
                        "logical": all_guidelines.get("logical", "")
                    }
                },
                {
                    "model": "gemini-2.5-flash",
                    "guidelines": {
                        "thinking": all_guidelines.get("thinking", ""),
                        "thought": all_guidelines.get("thought", "")
                    }
                },
                {
                    "model": "gemini-1.5-pro",
                    "guidelines": {
                        "tools": all_guidelines.get("tools", ""),
                        "structure": all_guidelines.get("structure", "")
                    }
                }
            ]
            status_placeholder.info("✅ Batches configured.")

            status_placeholder.info("[3/5] Kicking off batched parallel reviews...")
            all_findings, results_queue, threads = [], Queue(), []
            for batch in batches:
                thread = threading.Thread(target=run_batched_review_task, args=(results_queue, structured_notebook, batch["model"], batch["guidelines"]))
                threads.append(thread); thread.start()
                time.sleep(1) # Stagger API calls
            for thread in threads: thread.join()
            status_placeholder.info("✅ All review threads finished.")

            errors_found = []
            while not results_queue.empty():
                result_list = results_queue.get()
                for result in result_list:
                    if isinstance(result, dict) and result.get("error"):
                        errors_found.append(result)
                    else:
                        all_findings.append(result)
            
            if errors_found:
                return None, None, errors_found

            status_placeholder.info(f"[4/5] Running final validation with {VALIDATION_MODEL}...")
            if not all_findings:
                validation_result = {"final_report": [], "overall_feedback": "Excellent! No issues were found by any of the specialized auditors."}
            else:
                validation_prompt = build_validation_prompt(structured_notebook, all_findings)
                response = call_gemini_api(prompt=validation_prompt, system_prompt="You are the Senior AI Audit Validator...", model=VALIDATION_MODEL)
                validation_result = extract_json_from_response(response) if response else None
                if not validation_result:
                     validation_result = {"final_report": [], "overall_feedback": "Validation step failed."}
            
            status_placeholder.info("[5/5] Generating final report...")
            final_report = generate_final_report(validation_result, notebook_name)
            report_filename = f"FINAL_AUDIT_REPORT_{notebook_name.replace('.ipynb', '')}.md"
            
            status_placeholder.success("🎉 Audit Complete!")
            return final_report, report_filename, []
        
        except Exception as e:
            return None, None, [{"error": True, "guideline": "Main Workflow", "model": "N/A", "message": str(e), "traceback": traceback.format_exc()}]

# --- STREAMLIT UI ---
if __name__ == "__main__":
    st.set_page_config(page_title="AI Notebook Auditor", layout="wide")
    st.title("🤖 AI Notebook Auditor (Cost Optimized)")
    st.markdown("Enter a task number to perform a batched, multi-model review.")
    
    task_number = st.text_input("Enter the Task Number:", placeholder="e.g., 214514")

    if st.button("Start Review", type="primary", use_container_width=True):
        if task_number and task_number.isdigit():
            console_container = st.expander("Live Console Log", expanded=True)
            status_placeholder = console_container.empty()
            with st.spinner("Executing cost-optimized audit..."):
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

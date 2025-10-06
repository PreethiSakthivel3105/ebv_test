import os
import re
import json
import pandas as pd
import logging
import traceback
import requests
import httpx
import uuid
import time
import concurrent.futures

from pathlib import Path
from mistralai.models.sdkerror import SDKError
from mistralai import Mistral
from mistralai.models import DocumentURLChunk
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from io import BytesIO
from config import (
    PDF_FOLDER, PROCESS_COUNT, MISTRAL_API_KEY, bedrock,
    MISTRAL_OCR_COST_PER_1K_PAGES, BEDROCK_COST_PER_1K_TOKENS, LLM_PAGE_WORKERS,
    MAX_RETRIES, BACKOFF_MULTIPLIER, CLIENT_TIMEOUT, CONNECT_TIMEOUT
)
from database import get_db_connection, get_cached_result, cache_result, update_plan_file_hash, insert_acronyms_to_ref_table
from utils import (
    similarity, clean_drug_name, detect_prior_authorization,
    detect_step_therapy, calculate_file_hash, rate_limited_api_call,
    track_bedrock_cost_precalculated, track_mistral_cost, determine_coverage_status,
    normalize_drug_tier, infer_drug_tier_from_text
)

logger = logging.getLogger(__name__)

CLAUDE_3_HAIKU_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

import re
import json
import logging

try:
    import json5
    JSON5_AVAILABLE = True
except ImportError:
    JSON5_AVAILABLE = False


def robust_json_repair(json_string: str):
    """
    Attempts to repair malformed JSON returned by the LLM.
    Always returns a dict with keys: drug_table, acronyms, tiers.
    """
    default_output = {"drug_table": [], "acronyms": [], "tiers": []}

    if not isinstance(json_string, str) or not json_string.strip():
        return default_output

    # Step 1: Keep only the JSON-like content
    json_string = re.sub(r'^[^{\[]+', '', json_string)
    json_string = re.sub(r'[^}\]]+$', '', json_string)

    # Step 2: Cleanup formatting issues
    json_string = re.sub(r'[\r\n]+', ' ', json_string)
    json_string = re.sub(r'\s+', ' ', json_string)
    json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
    json_string = re.sub(r'"\s*"', '", "', json_string)
    json_string = re.sub(r'}\s*{', '}, {', json_string)
    json_string = re.sub(r'\]\s*\[', '], [', json_string)
    json_string = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', json_string)

    # Step 3: Balance braces/brackets
    open_curly, close_curly = json_string.count('{'), json_string.count('}')
    if open_curly > close_curly:
        json_string += '}' * (open_curly - close_curly)
    elif close_curly > open_curly:
        json_string = '{' * (close_curly - open_curly) + json_string

    open_sq, close_sq = json_string.count('['), json_string.count(']')
    if open_sq > close_sq:
        json_string += ']' * (open_sq - close_sq)
    elif close_sq > open_sq:
        json_string = '[' * (close_sq - open_sq) + json_string

    # Step 4: Try JSON5 if available
    if JSON5_AVAILABLE:
        try:
            parsed = json5.loads(json_string)
            return _sanitize_output(parsed, default_output)
        except Exception:
            pass

    # Step 5: Try standard JSON
    try:
        parsed = json.loads(json_string)
        return _sanitize_output(parsed, default_output)
    except json.JSONDecodeError:
        # Step 6: Last-resort comma fix and retry
        json_string = re.sub(r'"\s*([{\[])', '", \1', json_string)
        json_string = re.sub(r'([}\]])\s*"', r'\1, "', json_string)
        try:
            parsed = json.loads(json_string)
            return _sanitize_output(parsed, default_output)
        except json.JSONDecodeError as e2:
            logging.error(f"JSON repair failed: {e2}")
            logging.debug(f"Problematic JSON snippet: {json_string[:400]}...")
            return default_output


def _sanitize_output(parsed, default_output):
    """
    Ensures output always has expected keys.
    """
    if not isinstance(parsed, dict):
        return default_output

    sanitized = {
        "drug_table": parsed.get("drug_table", []),
        "acronyms": parsed.get("acronyms", []),
        "tiers": parsed.get("tiers", []),
    }
    return sanitized


def extract_metadata_from_filename(filename):
    """Extract state, payer, and plan name from filename"""
    base = os.path.splitext(filename)[0]
    parts = base.split("_", 2)
    if len(parts) != 3:
        logger.error(f"Filename format incorrect: {filename}. Expected State_Payer_Plan.")
        raise ValueError(f"Filename format incorrect: {filename}")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


@rate_limited_api_call
def extract_structured_data_with_llm(page_markdown: str):
    """
    Uses Claude 3 Haiku to parse markdown, correct multi-line entries,
    and extract structured drug data, requirement codes, and tier definitions.
    """
    costs = {'tokens': 0, 'cost': 0.0, 'calls': 1}
    if not bedrock:
        logger.error("Bedrock client is not initialized. Cannot extract structured data.")
        return {"drug_table": [], "acronyms": [], "tiers": []}, costs

    system_prompt = """
You are a highly specialized data extraction agent for pharmaceutical formularies. Your task is to meticulously extract three types of information from the provided page markdown: the main drug table, definitions for requirement codes (acronyms), and definitions for drug tiers.

From the provided page, you must extract:
1.  **Drug Formulary Requirement Codes**: Abbreviations in drug tables denoting requirements like Prior Authorization (e.g., PA, QL, ST). Find these in a "Key" or "Requirements/Limits" legend.
2.  **Drug Tier Definitions**: Definitions for cost-sharing tiers (e.g., Tier 1, Preferred Brand). Find these in a "Drug tier" section.
3.  **Drug Table Data**: The main drug list with columns: `drug_name`, `drug_tier`, `drug_requirements`.

**CRITICAL RULES:**
1.  **Extract ONLY from Explicit Definitions**: Extract acronyms and tiers ONLY if they are explicitly defined in a legend, key, glossary, or tier definition table on the page. If a section is missing, return its key with an empty list.
2.  **Strict Structure**: Your output MUST be a single JSON object with three keys: `drug_table`, `acronyms`, and `tiers`. If a section is missing, return its key with an empty list.
3.  **English Only**: All extracted data must be in English.

- **ACRONYMS (Formulary Requirement Codes)**:
    - **ONLY extract formulary requirement codes** (e.g., PA, QL, ST, MO, LD, B/D).
    - **DO NOT** extract plan types (HMO, PPO), dosage forms (HCL, ER), or general abbreviations (FDA).
    - **CRITICAL**: The `acronyms` list **MUST NOT** contain any tier definitions or any Numbers.
   - **DO NOT** include "Tier 1", "Tier 2", "Specialty Tier", "ACA", or plain numbers like "2", "3", "4". These are tiers.

- **TIERS (Tier Definitions)**:
    - **ONLY extract English drug tier definitions** (e.g., Tier 1, Tier 2, Preferred, Specialty, ACA).
    - **CRITICAL**: **DO NOT** extract requirement codes like PA, QL, MO, or B/D into the `tiers` list. These belong ONLY in the `acronyms` list.
    **EXTRACTION DETAILS:**

**1. `drug_table` (List of Objects):**
   - Extract the main list of drugs.
   - Each object in the list must have three keys: `drug_name`, `drug_tier`, `drug_requirements`.

      Primary columns are "Drug name", "Drug tier", and "Requirements/Limits".
    
    Few Variations [example]:
    drug_name: Match headers like: Drug Name, Medication, Brand Name, Generic Name, Formulary Drug, Product Name.
    drug_tier: Match headers like: Tier, Drug Tier, Formulary Tier, Cost Tier, Tier Level.
    drug_requirements: Match headers like: Requirements, Limits, Restrictions, Notes, PA, Prior Authorization, Step Therapy, QL, Quantity Limits, ST.
    
    Use keys: "drug_name", "drug_tier", "drug_requirements".

    CRITICAL:
    - Merge multiline drug names into a single drug_name.
    - If a row has a drug tier value in a separate column (e.g., "Tier 1", "TIER 2", "1", "T1"), populate "drug_tier" with normalized values "Tier 1", "Tier 2", "Tier 3", "Tier 4", or null if unknown.
    - drug_requirements holds prior auth / QL / PA details.
    - Ignore section headers (ALL CAPS).

    VERY VERY IMPORTANT NOTE---> INDEX PAGE DETECTION:
    - If you detect this is an INDEX or TABLE OF CONTENTS page (containing patterns like "Drug Name.......41", "Medication..........123", or lines with drug names followed by dots and page numbers), DO NOT extract any data.
    - INDEX INDICATORS: Lines with drug/medication names followed by multiple dots/periods and ending with numbers (page references).
    - If index content is detected, return an "empty" JSON array.

    Return ONLY a JSON array of objects with keys drug_name, drug_tier, drug_requirements.


**2. `acronyms` (Formulary Requirement Codes):**
   - Find the "Key", "Legend", or "Glossary" section that defines requirement codes.
   - **MUST contain ONLY formulary requirement codes.** Examples: PA, QL, ST, MO, B/D, ED, LA.
   - **CRITICAL**: The `acronyms` list **MUST NOT** contain any tier definitions.
   - **DO NOT** include "Tier 1", "Tier 2", "Specialty Tier", "ACA", or plain numbers like "2", "3", "4". These are tiers.
   - **DO NOT** extract plan types (HMO, PPO), dosage forms (HCL, ER), or general abbreviations (FDA).
   - Each object must have three keys:
     - `acronym`: The abbreviation (e.g., "PA", "QL", "ST").
     - `expansion`: The full name (e.g., "Prior Authorization", "Quantity Limit").
     - `explanation`: The detailed description or definition paragraph.

   - **Example:**
     ```json
     {
       "acronym": "PA",
       "expansion": "Prior Authorization",
       "explanation": "The plan requires you or your physician to get prior authorization for certain drugs. This means that you will need to get approval before you fill your prescriptions."
     }
     ```

**3. `tiers` (Drug Tier Definitions):**
   - Find the section that defines the drug tiers.
   - **MUST contain ONLY drug tier definitions.**
   - A tier is typically "Tier" followed by a number (Tier 1, Tier 2), a number by itself (2, 3, 4), or a name (Specialty Tier, Preferred, ACA).
   - **CRITICAL**: The `tiers` list **MUST NOT** contain formulary requirement codes like PA, QL, ST, MO. These belong in the `acronyms` list.
   - **DO NOT** extract tiers that are just numbers without a definition (e.g., if you see "Tier 27" in a drug table but there is no definition for it, ignore it). Tiers are typically numbered 1-6.
   - Each object must have three keys:
     - `acronym`: The tier identifier (e.g., "Tier 1", "Tier 2", "ACA").
     - `expansion`: The tier's name or type of drug (e.g., "Generic", "Preferred Drugs", "Non-Preferred Drug").
     - `explanation`: The detailed description of what the tier includes.
   - **Example:**
     ```json
     {
       "acronym": "Tier 1",
       "expansion": "Generic",
       "explanation": "Lowest-cost tier. Most generic drugs on the formulary are included in this tier."
     }
     ```
Return a JSON object with three keys: `drug_table`, `acronyms`, and `tiers`.
Extract data and return ONLY valid JSON with no additional text.

Format:
{"drug_table": [...], "acronyms": [...], "tiers": [...]}
If a section is missing, return its key with an empty list. Example: `{"drug_table": [], "acronyms": [], "tiers": []}`


Strictly Ensure:
- All strings are properly quoted
- No trailing commas
- Commas between all array/object elements
- All quotes are properly escaped.
"""
    user_message = f"<INPUT_MARKDOWN>\n{page_markdown}\n</INPUT_MARKDOWN>"

    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": 4096,
            "system": system_prompt, "messages": [{"role": "user", "content": user_message}]
        })
        response = bedrock.invoke_model(body=body, modelId=CLAUDE_3_HAIKU_MODEL_ID)
        response_body = json.loads(response.get('body').read())
        response_text = response_body['content'][0]['text']
        usage = response_body['usage']
        total_tokens = usage['input_tokens'] + usage['output_tokens']
        costs['tokens'] = total_tokens
        costs['cost'] = (total_tokens / 1000.0) * BEDROCK_COST_PER_1K_TOKENS

        # More robust JSON extraction
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL) or \
                     re.search(r'(\{.*?\})', response_text, re.DOTALL)

        if not json_match:
            logger.warning("LLM did not return a valid JSON object.")
            return {"drug_table": [], "acronyms": [], "tiers": []}, costs

        json_string = json_match.group(1)
        try:
            # More robust JSON repair:
            # 1. Remove trailing commas from objects and lists
            json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
            
            # 2. Add missing commas between adjacent objects/arrays (MOST COMMON ERROR)
            json_string = re.sub(r'}\s*{', r'},{', json_string)
            json_string = re.sub(r']\s*{', r'],{', json_string)
            json_string = re.sub(r'}\s*\[', r'},[', json_string)
            json_string = re.sub(r']\s*\[', r'],[', json_string)
            
            #structured_data = json.loads(json_string, strict=False)
            structured_data = robust_json_repair(json_string)
            
        except json.JSONDecodeError as e2:
            logger.error(f"JSON repair failed. Error: {e2}")
            logger.debug(f"Problematic JSON snippet: {json_string[:500]}...")
            return {"drug_table": [], "acronyms": [], "tiers": []}, costs

        logger.info(f"Successfully extracted {len(structured_data.get('drug_table', []))} drug records, {len(structured_data.get('acronyms', []))} acronyms, and {len(structured_data.get('tiers', []))} tiers.")
        
        # This block now defensively handles cases where the LLM returns a list of strings
        # instead of a list of dictionaries for 'acronyms' and 'tiers', which prevents the crash.
        blocklist = {'nivel'}
        for key in ['acronyms', 'tiers']:
            # Ensure the key exists and its value is a list before iterating
            if key in structured_data and isinstance(structured_data[key], list):
                filtered_list = []
                for item in structured_data[key]:
                    # Case 1: The item is a dictionary (the correct format)
                    if isinstance(item, dict):
                        acronym = str(item.get('acronym') or '').lower()
                        expansion = str(item.get('expansion') or '').lower()
                        if acronym not in blocklist and expansion not in blocklist:
                            filtered_list.append(item)
                    # Case 2: The item is a string (the error-causing format)
                    elif isinstance(item, str):
                        if item.lower() not in blocklist:
                            # Convert the string to the expected dictionary format to prevent data loss
                            logger.warning(f"LLM returned a string '{item}' in list '{key}'. Converting to dict.")
                            filtered_list.append({'acronym': item, 'expansion': None, 'explanation': None})
                    # Malformed data of other types will be safely ignored
                
                # Replace the original, potentially malformed list with the clean one
                structured_data[key] = filtered_list

        return structured_data, costs

    except Exception as e:
        logger.error(f"Error in Claude 3 Haiku LLM data extraction: {e}")
        traceback.print_exc()
        return {"drug_table": [], "acronyms": [], "tiers": []}, costs


def create_resilient_mistral_client():
    """
    Creates a Mistral client with robust timeouts and retry logic to prevent
    'Server disconnected' errors during large file uploads.
    """
    timeout = httpx.Timeout(CLIENT_TIMEOUT, connect=CONNECT_TIMEOUT)
    # The transport adapter handles the retry logic for specific HTTP errors
    transport = httpx.HTTPTransport(retries=MAX_RETRIES)
    client = httpx.Client(timeout=timeout, transport=transport)
    return Mistral(api_key=MISTRAL_API_KEY, client=client)


def process_pdf_with_mistral_ocr(pdf_input, payer_name=None):
    """
    Processes a PDF (from a file path or a BytesIO object) using Mistral OCR 
    and a parallelized LLM pipeline for data extraction, with robust retry logic.
    """
    log_name = getattr(pdf_input, 'name', pdf_input) if not isinstance(pdf_input, str) else pdf_input
    logger.info(f"Analyzing PDF with parallel LLM pipeline: {log_name}")

    total_costs = {'mistral_pages': 0, 'mistral_cost': 0.0, 'bedrock_tokens': 0, 'bedrock_cost': 0.0, 'bedrock_calls': 0}
    
    # Use the resilient client for all API interactions
    mistral_client = create_resilient_mistral_client()

    try:
        if isinstance(pdf_input, BytesIO):
            file_bytes = pdf_input.getvalue()
            file_name = "temp_in_memory.pdf"
        else:
            pdf_file = Path(pdf_input)
            file_bytes = pdf_file.read_bytes()
            file_name = pdf_file.name

        uploaded_file = None
        # Manual retry loop for the initial upload for extra safety
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES} to upload '{file_name}' to Mistral...")
                uploaded_file = mistral_client.files.upload(
                    file={"file_name": file_name, "content": file_bytes},
                    purpose="ocr",
                )
                logger.info("File uploaded successfully to Mistral.")
                break
            except (SDKError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError) as e:
                if attempt < MAX_RETRIES - 1:
                    delay = BACKOFF_MULTIPLIER ** attempt
                    logger.warning(f"Network or Server error during upload: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to upload file to Mistral after {MAX_RETRIES} attempts due to a persistent network/server error.")
                    raise # Re-raise the exception to fail the worker

        if not uploaded_file:
            return pd.DataFrame(), "", total_costs, [], []

        signed_url = mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=120)
        ocr_response = mistral_client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=False
        )

        page_count = len(ocr_response.pages)
        total_costs['mistral_pages'] = page_count
        total_costs['mistral_cost'] = (page_count / 1000.0) * MISTRAL_OCR_COST_PER_1K_PAGES

        all_structured_data, all_acronyms, all_tiers, all_raw_pages = [], [], [], []

        logger.info(f"Processing {page_count} pages in parallel with up to {LLM_PAGE_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=LLM_PAGE_WORKERS) as executor:
            future_to_page = {executor.submit(extract_structured_data_with_llm, page.markdown): page_idx + 1 for page_idx, page in enumerate(ocr_response.pages)}
            # Collect raw markdown content separately to avoid race conditions
            for page in ocr_response.pages:
                 all_raw_pages.append(page.markdown)

            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    structured_records, llm_costs = future.result()
                    logger.info(f"--- Completed processing for Page {page_num}/{page_count} ---")
                    total_costs['bedrock_tokens'] += llm_costs.get('tokens', 0)
                    total_costs['bedrock_cost'] += llm_costs.get('cost', 0)
                    total_costs['bedrock_calls'] += llm_costs.get('calls', 0)
                    if structured_records:
                        all_structured_data.extend(structured_records.get('drug_table', []))
                        all_acronyms.extend(structured_records.get('acronyms', []))
                        all_tiers.extend(structured_records.get('tiers', []))
                except Exception as exc:
                    logger.error(f"Page {page_num} generated an exception during result processing: {exc}")

        full_raw_content = "\n\n--- PAGE BREAK ---\n\n".join(all_raw_pages)
        structured_df = pd.DataFrame(all_structured_data) if all_structured_data else pd.DataFrame()
        logger.info(f"Final results: {len(structured_df)} structured records extracted from PDF.")

        try:
            mistral_client.files.delete(file_id=uploaded_file.id)
            logger.info(f"Deleted uploaded file from Mistral: {uploaded_file.id}")
        except Exception as e:
            logger.warning(f"Failed to delete uploaded file {uploaded_file.id}: {e}")

        return structured_df, full_raw_content, total_costs, all_acronyms, all_tiers

    except Exception as e:
        logger.error(f"A critical error occurred in the main PDF processing pipeline for {log_name}: {e}")
        traceback.print_exc()
        # Return empty structures to prevent downstream errors
        return pd.DataFrame(), "", total_costs, [], []


def get_plan_and_payer_info(state_name, payer, plan_name):
    """Get plan_id and payer_id from database with exact and fuzzy matching."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            logger.info(f"Looking for: State='{state_name}', Payer='{payer}', Plan='{plan_name}'")
            exact_query = """
                SELECT pd.plan_id, pd.payer_id, py.payer_name, pd.plan_name, pd.formulary_url
                FROM plan_details pd JOIN payer_details py ON pd.payer_id = py.payer_id
                WHERE LOWER(TRIM(pd.state_name)) = LOWER(TRIM(%s))
                  AND LOWER(TRIM(py.payer_name)) = LOWER(TRIM(%s))
                  AND LOWER(TRIM(pd.plan_name)) ILIKE LOWER(TRIM(%s));
            """
            cursor.execute(exact_query, (state_name, payer, f'%{plan_name}%')) # Use ILIKE for plan name
            result = cursor.fetchone()
            if result:
                plan_id, payer_id, db_payer_name, db_plan_name, formulary_url = result
                logger.info(f"Found match in DB: Plan='{db_plan_name}', Payer='{db_payer_name}'")
                return plan_id, payer_id, db_payer_name, db_plan_name, formulary_url

            logger.warning(f"No exact match for '{plan_name}'. Falling back to fuzzy matching...")
            cursor.execute("""
                SELECT pd.plan_id, pd.payer_id, py.payer_name, pd.plan_name, pd.formulary_url
                FROM plan_details pd JOIN payer_details py ON pd.payer_id = py.payer_id
                WHERE LOWER(TRIM(pd.state_name)) = LOWER(TRIM(%s))
            """, (state_name,))
            all_records_in_state = cursor.fetchall()
            if not all_records_in_state:
                 logger.error(f"Fuzzy match failed: No plans found for state '{state_name}'")
                 return None, None, None, None, None

            best_match, best_score = None, 0.70 # Increased threshold
            for record in all_records_in_state:
                plan_id, payer_id, db_payer_name, db_plan_name, formulary_url = record
                payer_score = similarity(payer, db_payer_name)
                plan_score = similarity(plan_name, db_plan_name)
                total_score = (payer_score * 0.4) + (plan_score * 0.6)
                if total_score > best_score:
                    best_score = total_score
                    best_match = record

            if best_match:
                plan_id, payer_id, db_payer_name, db_plan_name, formulary_url = best_match
                logger.info(f"Found fuzzy match (score: {best_score:.2f}): Plan='{db_plan_name}', Payer='{db_payer_name}'")
                return plan_id, payer_id, db_payer_name, db_plan_name, formulary_url

            logger.error(f"Fuzzy match failed for plan '{plan_name}' in state '{state_name}'.")
            return None, None, None, None, None

        except Exception as e:
            logger.error(f"Error in get_plan_and_payer_info: {e}")
            return None, None, None, None, None

def deduplicate_dicts(dicts, primary_key='acronym'):
    """Deduplicates a list of dictionaries, merging to keep the most complete info."""
    if not dicts:
        return []
    merged_entries = {}
    for item in dicts:
        key_value = item.get(primary_key)
        if not key_value:
            continue
        key = str(key_value).strip().lower()
        if key not in merged_entries:
            merged_entries[key] = item.copy()
        else:
            current_best = merged_entries[key]
            for field in ['expansion', 'explanation']:
                new_value = item.get(field)
                if new_value and len(str(new_value)) > len(str(current_best.get(field) or '')):
                    current_best[field] = new_value
    return list(merged_entries.values())

# --- WORKER AND ORCHESTRATOR FOR LOCAL PDFS ---

def process_single_pdf_worker(filename: str, pdf_folder_path: str):
    """
    Worker function for processing a single local PDF file.
    Includes caching, data extraction, normalization, and record creation.
    """
    log_prefix = f"[Worker for {filename}]"
    zero_costs = {'mistral_pages': 0, 'bedrock_tokens': 0, 'bedrock_cost': 0.0, 'bedrock_calls': 0}

    try:
        full_path = os.path.join(pdf_folder_path, filename)
        if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:
            return 'ERROR', filename, "File not found or is empty.", zero_costs

        state_name, payer, plan_name = extract_metadata_from_filename(filename)
        plan_id, payer_id, db_payer_name, db_plan_name, formulary_url = get_plan_and_payer_info(state_name, payer, plan_name)
        if not plan_id:
            return 'SKIPPED', filename, f"Plan not found in DB for: {state_name}, {payer}, {plan_name}", zero_costs

        file_hash = calculate_file_hash(full_path)
        update_plan_file_hash(plan_id, file_hash)

        structured_df, raw_content = get_cached_result(file_hash)
        costs = zero_costs
        if structured_df is None or structured_df.empty:
            logger.info(f"{log_prefix} Cache MISS. Starting full processing...")
            structured_df, raw_content, costs, all_acronyms, all_tiers = process_pdf_with_mistral_ocr(full_path, db_payer_name)
            
            all_acronyms, all_tiers = _reclassify_definitions(all_acronyms, all_tiers)
            logger.info(f"Re-classified definitions. Final counts: {len(all_acronyms)} acronyms, {len(all_tiers)} tiers.")

            all_tiers = _parse_and_split_tier_definitions(all_tiers)


            for tier_dict in all_tiers:
                acronym = tier_dict.get('acronym')
                if acronym and str(acronym).strip().isdigit():
                    tier_dict['acronym'] = f"Tier {str(acronym).strip()}"

            dedup_acronyms = deduplicate_dicts(all_acronyms)
            dedup_tiers = deduplicate_dicts(all_tiers)

            if dedup_acronyms:
                insert_acronyms_to_ref_table(dedup_acronyms, state_name, payer, plan_name, "PP_Formulary_Short_Codes_Ref")
            if dedup_tiers:
                insert_acronyms_to_ref_table(dedup_tiers, state_name, payer, plan_name, "PP_Tier_Codes_Ref")

            if not structured_df.empty:
                cache_result(file_hash, structured_df, raw_content)
        else:
             logger.info(f"{log_prefix} Cache HIT. Using pre-processed data.")

        if structured_df.empty:
            return 'SKIPPED', filename, "No structured data could be extracted.", costs

        processed_records = []
        for _, row in structured_df.iterrows():
            try:
                raw_drug_name = str(row.get('drug_name', '') or '')
                requirements_text = str(row.get('drug_requirements', '') or '').strip()
                cleaned_drug_name = clean_drug_name(raw_drug_name)
                if not cleaned_drug_name: continue

                raw_tier = row.get('drug_tier', None)
                drug_tier_normalized = normalize_drug_tier(raw_tier) or infer_drug_tier_from_text(requirements_text) or infer_drug_tier_from_text(raw_drug_name)

                with get_db_connection() as conn:
                    coverage_status = determine_coverage_status(requirements_text, drug_tier_normalized, conn, state_name, db_payer_name)

                record = {
                    "id": str(uuid.uuid4()), "plan_id": plan_id, "payer_id": payer_id,
                    "drug_name": cleaned_drug_name, "state_name": state_name,
                    "coverage_status": coverage_status, "drug_tier": drug_tier_normalized,
                    "drug_requirements": requirements_text or None,
                    "is_prior_authorization_required": "Yes" if detect_prior_authorization(requirements_text) else "No",
                    "is_step_therapy_required": "Yes" if detect_step_therapy(requirements_text) else "No",
                    "is_quantity_limit_applied": "Yes" if "ql" in (requirements_text or "").lower() else "No",
                    "confidence_score": 0.95, "source_url": formulary_url,
                    "plan_name": db_plan_name, "payer_name": db_payer_name, "file_name": filename,
                    "ndc_code": None, "jcode": None, "coverage_details": None,
                }
                processed_records.append(record)
            except Exception as e:
                logger.warning(f"{log_prefix} Error processing extracted row: {row}. Error: {e}")
                continue

        if processed_records:
            return 'SUCCESS', filename, {"processed_records": processed_records, "db_payer_name": db_payer_name}, costs
        else:
            return 'SKIPPED', filename, "Data extracted, but no valid drug records could be processed.", costs

    except Exception as e:
        return 'ERROR', filename, f"An unexpected error occurred in worker: {e}\n{traceback.format_exc()}", zero_costs


def process_pdfs_in_parallel():
    """Processes all PDFs in a local folder in parallel using a ProcessPoolExecutor."""
    logger.info("STEP 2: Processing Local PDF Files in Parallel")
    all_processed_data = []
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"No PDF files found in '{PDF_FOLDER}'.")
        return [], {}

    # Define a generous timeout for each PDF file in seconds (e.g., 20 minutes)
    PDF_PROCESSING_TIMEOUT = 1200 

    logger.info(f"Found {len(pdf_files)} PDFs. Starting parallel processing with up to {PROCESS_COUNT} workers.")
    success_count, error_count, skipped_count = 0, 0, 0
    with ProcessPoolExecutor(max_workers=PROCESS_COUNT) as executor:
        future_to_filename = {executor.submit(process_single_pdf_worker, filename, PDF_FOLDER): filename for filename in pdf_files}
        for future in as_completed(future_to_filename):
            filename = future_to_filename[future]
            try:
                # Wait for the result, but no longer than the timeout
                status, _, result_data, costs = future.result(timeout=PDF_PROCESSING_TIMEOUT)
                
                if status == 'SUCCESS':
                    success_count += 1
                    payer_name = result_data['db_payer_name']
                    if costs['mistral_pages'] > 0:
                        track_mistral_cost(payer_name, costs['mistral_pages'])
                    if costs['bedrock_tokens'] > 0:
                        track_bedrock_cost_precalculated(payer_name, costs['bedrock_tokens'], costs['bedrock_cost'], costs['bedrock_calls'])
                    all_processed_data.extend(result_data["processed_records"])
                elif status == 'SKIPPED':
                    skipped_count += 1
                    logger.warning(f"Skipped file: {filename}. Reason: {result_data}")
                elif status == 'ERROR':
                    error_count += 1
                    logger.error(f"Error processing file: {filename}. Reason: {result_data}")

            except concurrent.futures.TimeoutError:
                error_count += 1
                logger.error(f"CRITICAL: Processing timed out for file: {filename} after {PDF_PROCESSING_TIMEOUT} seconds. The worker is likely stuck. Moving on.")
            except Exception as e:
                error_count += 1
                logger.error(f"Critical error processing result for {filename}: {e}", exc_info=True)

    logger.info("--- Local PDF Processing Complete ---")
    logger.info(f"Summary: {success_count} successful, {error_count} failed, {skipped_count} skipped")
    logger.info(f"Total structured records aggregated: {len(all_processed_data)}")
    return all_processed_data, {}


# --- WORKER AND ORCHESTRATOR FOR URLS ---

def get_all_plans_with_formulary_url():
    """Fetch all plans marked 'processing' with a non-null formulary_url."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pd.state_name, py.payer_name, pd.plan_name, pd.plan_id, py.payer_id, pd.formulary_url
            FROM plan_details pd JOIN payer_details py ON pd.payer_id = py.payer_id
            WHERE pd.formulary_url IS NOT NULL AND pd.formulary_url != '' AND pd.status = 'processing'
        """)
        return cursor.fetchall()


def process_single_pdf_url_worker(plan_info):
    """Worker: Download PDF from URL and process it entirely in-memory."""
    state_name, payer_name, plan_name, plan_id, payer_id, formulary_url = plan_info
    log_prefix = f"[URL Worker for {plan_name}]"
    zero_costs = {'mistral_pages': 0, 'bedrock_tokens': 0, 'bedrock_cost': 0.0, 'bedrock_calls': 0}
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
        with requests.get(formulary_url, timeout=90, headers=headers, stream=True) as resp:
            resp.raise_for_status()
            if 'application/pdf' not in resp.headers.get('Content-Type', ''):
                return 'ERROR', plan_name, f"Invalid content type: {resp.headers.get('Content-Type', '')}", zero_costs
            pdf_bytes = BytesIO(resp.content)

        structured_df, raw_content, costs, all_acronyms, all_tiers = process_pdf_with_mistral_ocr(pdf_bytes, payer_name)

        all_acronyms, all_tiers = _reclassify_definitions(all_acronyms, all_tiers)
        logger.info(f"Re-classified definitions. Final counts: {len(all_acronyms)} acronyms, {len(all_tiers)} tiers.")

        all_tiers = _parse_and_split_tier_definitions(all_tiers)

        for tier_dict in all_tiers:
            acronym = tier_dict.get('acronym')
            if acronym and str(acronym).strip().isdigit():
                tier_dict['acronym'] = f"Tier {str(acronym).strip()}"

        dedup_acronyms = deduplicate_dicts(all_acronyms)
        dedup_tiers = deduplicate_dicts(all_tiers)

        if dedup_acronyms:
            insert_acronyms_to_ref_table(dedup_acronyms, state_name, payer_name, plan_name, "PP_Formulary_Short_Codes_Ref")
        if dedup_tiers:
            insert_acronyms_to_ref_table(dedup_tiers, state_name, payer_name, plan_name, "PP_Tier_Codes_Ref")

        if structured_df.empty:
            return 'SKIPPED', plan_name, "No structured data extracted from URL PDF.", costs

        processed_records = []
        for _, row in structured_df.iterrows():
            cleaned_drug_name = clean_drug_name(str(row.get('drug_name', '') or ''))
            if not cleaned_drug_name: continue
            requirements_text = str(row.get('drug_requirements', '') or '').strip()
            drug_tier_normalized = normalize_drug_tier(row.get('drug_tier', None)) or infer_drug_tier_from_text(requirements_text) or infer_drug_tier_from_text(cleaned_drug_name)

            with get_db_connection() as conn:
                coverage_status = determine_coverage_status(requirements_text, drug_tier_normalized, conn, state_name, payer_name)

            record = {
                "id": str(uuid.uuid4()), "plan_id": plan_id, "payer_id": payer_id,
                "drug_name": cleaned_drug_name, "state_name": state_name, "coverage_status": coverage_status,
                "drug_tier": drug_tier_normalized, "drug_requirements": requirements_text or None,
                "is_prior_authorization_required": "Yes" if detect_prior_authorization(requirements_text) else "No",
                "is_step_therapy_required": "Yes" if detect_step_therapy(requirements_text) else "No",
                "is_quantity_limit_applied": "Yes" if "ql" in (requirements_text or "").lower() else "No",
                "confidence_score": 0.95, "source_url": formulary_url,
                "plan_name": plan_name, "payer_name": payer_name,
                "file_name": f"{state_name}_{payer_name}_{plan_name}.pdf",
                # Add missing fields for DB insertion
                "ndc_code": None, "jcode": None, "coverage_details": None,
            }
            processed_records.append(record)

        if processed_records:
            return 'SUCCESS', plan_name, {"processed_records": processed_records, "db_payer_name": payer_name}, costs
        else:
            return 'SKIPPED', plan_name, "Data extracted, but no valid drug records were processed.", costs

    except Exception as e:
        logger.error(f"{log_prefix} Error: {e}", exc_info=True)
        return 'ERROR', plan_name, str(e), zero_costs


def process_pdfs_from_urls_in_parallel():
    """Process PDFs by downloading from URLs in plan_details, in parallel."""
    logger.info("STEP 2: Processing PDF Files from URLs in plan_details")
    all_processed_data = []

    plans = get_all_plans_with_formulary_url()
    if not plans:
        logger.warning("No plans with formulary URLs found to process.")
        return [], {}

    URL_PROCESSING_TIMEOUT = 1200

    logger.info(f"Found {len(plans)} plans with URLs to process.")
    success_count, error_count, skipped_count = 0, 0, 0
    with ProcessPoolExecutor(max_workers=PROCESS_COUNT) as executor:
        future_to_plan = {executor.submit(process_single_pdf_url_worker, plan): plan for plan in plans}

        for future in as_completed(future_to_plan):
            plan_info = future_to_plan[future]
            plan_name_log = plan_info[2] # Get plan name for logging
            try:
                # Wait for the result, but no longer than the timeout
                status, _, result_data, costs = future.result(timeout=URL_PROCESSING_TIMEOUT)
                
                if status == 'SUCCESS':
                    logger.info(f"Aggregating results for SUCCESSFUL plan: {plan_name_log}")
                    success_count += 1
                    payer_name = result_data['db_payer_name']
                    if costs['mistral_pages'] > 0:
                        track_mistral_cost(payer_name, costs['mistral_pages'])
                    if costs['bedrock_tokens'] > 0:
                        track_bedrock_cost_precalculated(payer_name, costs['bedrock_tokens'], costs['bedrock_cost'], costs['bedrock_calls'])
                    all_processed_data.extend(result_data["processed_records"])
                elif status == 'SKIPPED':
                    logger.warning(f"Skipped plan: {plan_name_log}. Reason: {result_data}")
                    skipped_count += 1
                elif status == 'ERROR':
                    logger.error(f"Error processing plan: {plan_name_log}. Reason: {result_data}")
                    error_count += 1
            
            except concurrent.futures.TimeoutError:
                error_count += 1
                logger.error(f"CRITICAL: Processing timed out for plan: {plan_name_log} after {URL_PROCESSING_TIMEOUT} seconds. The worker is likely stuck. Moving on.")
            except Exception as e:
                logger.error(f"A critical error occurred while processing result for {plan_name_log}: {e}", exc_info=True)
                error_count += 1

    logger.info("--- URL PDF Processing Complete ---")
    logger.info(f"Summary: {success_count} successful, {error_count} failed, {skipped_count} skipped")
    logger.info(f"Total structured records aggregated: {len(all_processed_data)}")
    return all_processed_data, {}


def deduplicate_dicts(dicts, primary_key='acronym'):
    """
    Deduplicates a list of dictionaries based on a primary key, merging them by keeping the most complete information.
    It prioritizes entries with longer 'expansion' and 'explanation' fields.
    """
    if not dicts:
        return []

    merged_entries = {}
    for item in dicts:
        key_value = item.get(primary_key)
        if not key_value:
            continue
        
        # Use a normalized key for merging (lowercase, stripped)
        key = str(key_value).strip().lower()

        if key not in merged_entries:
            merged_entries[key] = item.copy() # Use a copy to avoid modifying the original list
        else:
            # An entry with this key already exists, let's merge.
            current_best = merged_entries[key]
            
            # Update expansion if the new one is better (longer or more descriptive)
            new_expansion = item.get('expansion')
            if new_expansion and len(str(new_expansion)) > len(str(current_best.get('expansion') or '')):
                current_best['expansion'] = new_expansion
            
            # Update explanation if the new one is better (longer)
            new_explanation = item.get('explanation')
            if new_explanation and len(str(new_explanation)) > len(str(current_best.get('explanation') or '')):
                current_best['explanation'] = new_explanation
    
    return list(merged_entries.values())


def _reclassify_definitions(acronyms_list: list, tiers_list: list) -> (list, list):
    """
    Sorts definitions into acronyms or tiers based on heuristics to correct LLM misclassifications.
    This is a post-processing step to improve data quality without an extra LLM call.
    """
    if not tiers_list and not acronyms_list:
        return [], []

    corrected_acronyms = []
    corrected_tiers = []
    
    # Define keywords that identify a definition as a tier, even if it doesn't start with "Tier".
    TIER_KEYWORDS = {'aca', 'preventive', 'specialty', 'preferred', 'generic', 'brand'}

    # First, check the list of items the LLM classified as tiers.
    for item in tiers_list:
        if not isinstance(item, dict): continue
        acronym = str(item.get('acronym') or '').strip().lower()

        # Rule: If it starts with "tier" or is a known tier keyword, it's a tier. Otherwise, it's a misclassified formulary code.
        if acronym.startswith('tier') or acronym in TIER_KEYWORDS:
            corrected_tiers.append(item)
        else:
            # This item is likely a formulary code, move it to the correct list.
            corrected_acronyms.append(item)
            
    # Second, process the list the LLM already identified as acronyms.
    # This loop is mostly for completeness and to catch edge cases.
    for item in acronyms_list:
        if not isinstance(item, dict): continue
        acronym = str(item.get('acronym') or '').strip().lower()

        # Rule: If an item in the acronyms list actually looks like a tier, move it. This is rare.
        if acronym.startswith('tier') or acronym in TIER_KEYWORDS:
            corrected_tiers.append(item)
        else:
            corrected_acronyms.append(item)

    return corrected_acronyms, corrected_tiers


def _parse_and_split_tier_definitions(tier_list: list) -> list:
    """
    Parses tier definitions where the acronym and expansion might be combined in one field.
    This corrects LLM outputs like {"acronym": "Tier 1 - Generic", "expansion": None}
    into {"acronym": "Tier 1", "expansion": "Generic"}.
    """
    if not tier_list:
        return []

    processed_tiers = []
    for tier_dict in tier_list:
        # Ensure we're working with a dictionary, not a malformed string
        if not isinstance(tier_dict, dict):
            continue

        acronym_raw = tier_dict.get('acronym')
        expansion_raw = tier_dict.get('expansion')

        # Check if the acronym field is a string and contains a clear separator
        if isinstance(acronym_raw, str) and ' - ' in acronym_raw:
            parts = acronym_raw.split(' - ', 1)
            new_acronym = parts[0].strip()
            new_expansion = parts[1].strip()
            
            # Update the dictionary with the separated values
            tier_dict['acronym'] = new_acronym
            
            # Only fill the expansion if it was originally empty to avoid overwriting good data
            if not expansion_raw:
                tier_dict['expansion'] = new_expansion
        
        processed_tiers.append(tier_dict)
        
    return processed_tiers
import os
import re
import json
import pandas as pd
import logging
import traceback
from pathlib import Path
from mistralai import Mistral
from mistralai.models import DocumentURLChunk
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import requests
from io import BytesIO

from config import mistral_client, PDF_FOLDER, PROCESS_COUNT, MISTRAL_API_KEY, bedrock, MISTRAL_OCR_COST_PER_1K_PAGES, BEDROCK_COST_PER_1K_TOKENS, LLM_PAGE_WORKERS
from database import get_db_connection, get_cached_result, cache_result, update_plan_file_hash, insert_acronyms_to_ref_table
import uuid
from utils import similarity, normalize_text, clean_drug_name, detect_prior_authorization, detect_step_therapy, calculate_file_hash, rate_limited_api_call, estimate_tokens, track_bedrock_cost, track_mistral_cost, extract_requirements_from_drug_name, determine_coverage_status, track_bedrock_cost_precalculated, normalize_drug_tier, infer_drug_tier_from_text

logger = logging.getLogger(__name__)

CLAUDE_3_HAIKU_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

def extract_metadata_from_filename(filename):
    """Extract state, payer, and plan name from filename"""
    base = os.path.splitext(filename)[0]
    parts = base.split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Filename format incorrect: {filename}")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


@rate_limited_api_call
def extract_structured_data_with_llm(page_markdown: str):
    """
    Uses the fast and cost-effective Claude 3 Haiku model to parse messy markdown,
    correcting multi-line entries and returning clean structured data.
    Now also extracts all abbreviations/acronyms and their expansions/explanations.
    """
    logger.info("Extracting structured data and acronyms from page markdown using Claude 3 Haiku...")

    costs = {'tokens': 0, 'cost': 0.0, 'calls': 1}
    
    system_prompt = """
You are a highly specialized data extraction agent for pharmaceutical formularies. Your task is to meticulously extract information ONLY from dedicated "legend", "key", or "glossary" sections on the page.

From the provided page, you must extract:
1.  **Drug Formulary Requirement Codes**: These are abbreviations used in the drug tables to denote requirements like Prior Authorization or Quantity Limits. They are typically found in a "Key" or "Requirements/Limits" legend.
2.  **Drug Tier Definitions**: These define the cost-sharing tiers for drugs (e.g., Tier 1, Preferred Brand). They are often found in a "Drug tier copay levels" or similar section.
3.  **Drug Table Data**: Extract the main drug list with columns: `drug_name`, `drug_tier`, `drug_requirements`.

**CRITICAL RULES:**
- **FOCUS ONLY ON LEGENDS**: Extract acronyms and tiers ONLY if they are explicitly defined in a legend, key, or glossary on the page. Ignore abbreviations found in running text, headers, or footers.
- **IGNORE NON-ENGLISH TEXT**: All extracted acronyms, expansions, and tiers must be in English. For example, DO NOT extract 'Nivel' as a tier.
- **ACRONYMS (Requirement Codes)**:
    - **ONLY extract formulary requirement codes** (e.g., PA, QL, ST, MO, LD, B/D, ACS, HRM).
    - **DO NOT** extract plan types (e.g., HMO, D-SNP, PPO, QMB).
    - **DO NOT** extract general medical or dosage abbreviations (e.g., HCL, ER, DR, ODT, EA, ML, FDA).
- **TIERS (Tier Definitions)**:
    - **ONLY extract English drug tier definitions** (e.g., Tier 1, Tier 2, Preferred, Non-Preferred, Specialty).
    - **DO NOT** include requirement codes (like MO, LD, QL) in the tiers list.

Return a JSON object with three keys:
- `"drug_table"`: An array of objects with keys `drug_name`, `drug_tier`, `drug_requirements`.
- `"acronyms"`: An array of objects ONLY for **formulary requirement codes**, with keys `acronym`, `expansion`, `explanation`.
- `"tiers"`: An array of objects ONLY for **drug tier definitions**, with keys `acronym`, `expansion`, `explanation`.

If a page has no relevant data, a key is missing, or a list is empty, return it as an empty list. Example: `{"drug_table": [], "acronyms": [], "tiers": []}`.

**Example of CORRECT Output:**
{
  "drug_table": [
    {"drug_name": "allopurinol tablet 100mg, 300mg", "drug_tier": null, "drug_requirements": "MO"}
  ],
  "acronyms": [
    {"acronym": "PA", "expansion": "Prior Authorization", "explanation": "Our plan requires you or your prescriber to get prior authorization..."},
    {"acronym": "QL", "expansion": "Quantity Limits", "explanation": "For certain drugs, our plan limits the amount of the drug..."}
  ],
  "tiers": [
    {"acronym": "Tier 1", "expansion": "Generic", "explanation": "Generic drugs are the low-cost version..."}
  ]
}
"""

    user_message = f"""
    --- Now process the following markdown and extract the required fields:
    <INPUT_MARKDOWN>
    {page_markdown}
    </INPUT_MARKDOWN>
    """
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}]
        })

        response = bedrock.invoke_model(body=body, modelId=CLAUDE_3_HAIKU_MODEL_ID)
        response_body = json.loads(response.get('body').read())
        
        response_text = response_body['content'][0]['text']
        usage = response_body['usage']
        
        total_tokens = usage['input_tokens'] + usage['output_tokens']
        costs['tokens'] = total_tokens
        costs['cost'] = (total_tokens / 1000.0) * BEDROCK_COST_PER_1K_TOKENS
        
        logger.debug(f"Claude 3 Haiku Response: {response_text}")

        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'```json\s*(\{.*\})\s*```', response_text, re.DOTALL)
            if not json_match:
                logger.warning("LLM did not return a valid JSON object.")
                return {"drug_table": [], "acronyms": [], "tiers": []}, costs

        json_string = json_match.group(1) if '```json' in json_match.group(0) else json_match.group(0)

        try:
            structured_data = json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.warning(f"Caught JSONDecodeError: {e}. Attempting a lenient parse...")
            repaired_json_string = json_string.replace('\\', '\\\\')
            try:
                structured_data = json.loads(repaired_json_string, strict=False)
                logger.info("Successfully repaired and parsed the JSON string.")
            except json.JSONDecodeError as e2:
                logger.error(f"JSON repair failed. The string is likely malformed. Error: {e2}")
                logger.debug(f"Problematic JSON snippet: {repaired_json_string[:500]}...")
                return {"drug_table": [], "acronyms": [], "tiers": []}, costs

        logger.info(f"Successfully extracted {len(structured_data.get('drug_table', []))} drug records, {len(structured_data.get('acronyms', []))} acronyms, and {len(structured_data.get('tiers', []))} tiers from page.")
        
        # <<< START OF MODIFICATION >>>
        # Post-processing to filter out unwanted terms and handle non-string data
        blocklist = {'nivel'}
        
        if 'acronyms' in structured_data:
            structured_data['acronyms'] = [
                ac for ac in structured_data.get('acronyms', [])
                # Convert to string before calling .lower() to prevent AttributeError
                if str(ac.get('acronym') or '').lower() not in blocklist and \
                   str(ac.get('expansion') or '').lower() not in blocklist
            ]

        if 'tiers' in structured_data:
            structured_data['tiers'] = [
                tier for tier in structured_data.get('tiers', [])
                # Convert to string before calling .lower() to prevent AttributeError
                if str(tier.get('acronym') or '').lower() not in blocklist and \
                   str(tier.get('expansion') or '').lower() not in blocklist
            ]
        # <<< END OF MODIFICATION >>>
        
        return structured_data, costs
    

    except Exception as e:
        logger.error(f"Error in Claude 3 Haiku LLM data extraction: {e}")
        traceback.print_exc()
        return {"drug_table": [], "acronyms": [], "tiers": []}, costs


def process_pdf_with_mistral_ocr(pdf_input, mistral_client, payer_name=None):
    """
    Accepts either a file path (str) or a BytesIO object for the PDF.
    """
    logger.info(f"Analyzing PDF with parallel LLM pipeline: {getattr(pdf_input, 'name', pdf_input) if not isinstance(pdf_input, str) else pdf_input}")

    total_costs = {'mistral_pages': 0, 'mistral_cost': 0.0, 'bedrock_tokens': 0, 'bedrock_cost': 0.0, 'bedrock_calls': 0}

    try:
        if isinstance(pdf_input, BytesIO):
            file_bytes = pdf_input.getvalue()
            file_name = "temp.pdf"
        else:
            pdf_file = Path(pdf_input)
            file_bytes = pdf_file.read_bytes()
            file_name = pdf_file.stem

        uploaded_file = mistral_client.files.upload(
            file={"file_name": file_name, "content": file_bytes},
            purpose="ocr",
        )
        signed_url = mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=60)
        ocr_response = mistral_client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=False
        )
        
        page_count = len(ocr_response.pages)
        total_costs['mistral_pages'] = page_count
        total_costs['mistral_cost'] = (page_count / 1000.0) * MISTRAL_OCR_COST_PER_1K_PAGES
        
        all_structured_data = []
        all_acronyms = []
        all_tiers = []
        all_raw_pages = []  

        # 2. Iterate through pages and use Claude 3 for structured extraction IN PARALLEL
        logger.info(f"Processing {page_count} pages in parallel with up to {LLM_PAGE_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=LLM_PAGE_WORKERS) as executor:
            futures = {}
            for page_idx, page in enumerate(ocr_response.pages):
                page_num = page_idx + 1
                markdown_content = page.markdown
                all_raw_pages.append(markdown_content) 

                if not markdown_content or len(markdown_content.strip()) < 50:
                    logger.info(f"Skipping LLM processing for page {page_num} due to insufficient content.")
                    continue
                
                future = executor.submit(extract_structured_data_with_llm, markdown_content)
                futures[future] = page_num

            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    structured_records, llm_costs = future.result()
                    logger.info(f"--- Completed processing for Page {page_num}/{page_count} ---")
                    
                    total_costs['bedrock_tokens'] += llm_costs.get('tokens', 0)
                    total_costs['bedrock_cost'] += llm_costs.get('cost', 0)
                    total_costs['bedrock_calls'] += llm_costs.get('calls', 0)
                    
                    if structured_records:
                        # Ensure keys exist before extending
                        all_structured_data.extend(structured_records.get('drug_table', []))
                        all_acronyms.extend(structured_records.get('acronyms', []))
                        all_tiers.extend(structured_records.get('tiers', []))
                except Exception as exc:
                    logger.error(f"Page {page_num} generated an exception during result processing: {exc}")

        # 3. Combine all raw markdown and structured results
        full_raw_content = "\n\n--- PAGE BREAK ---\n\n".join(all_raw_pages) 
        
        structured_df = pd.DataFrame()
        if all_structured_data:
            structured_df = pd.DataFrame(all_structured_data)
        
        logger.info(f"Final results: {len(structured_df)} structured records extracted from PDF.")
        
        try:
            mistral_client.files.delete(file_id=uploaded_file.id)
            logger.info(f"Deleted uploaded file from Mistral: {uploaded_file.id}")
        except Exception as e:
            logger.warning(f"Failed to delete uploaded file: {e}")
         
        return structured_df, full_raw_content, total_costs, all_acronyms, all_tiers

    except Exception as e:
        logger.error(f"Error in main PDF processing pipeline: {e}")
        traceback.print_exc()
        return pd.DataFrame(), "", total_costs, [], []

def get_plan_and_payer_info(state_name, payer, plan_name):
    """Get plan_id and payer_id from database with a more robust, query-based approach."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            logger.info(f"Looking for: State='{state_name}', Payer='{payer}', Plan='{plan_name}'")
            
            normalized_payer = payer
            normalized_plan_name = plan_name
            exact_query = """
                SELECT pd.plan_id, pd.payer_id, py.payer_name, pd.plan_name, pd.formulary_url
                FROM plan_details pd
                JOIN payer_details py ON pd.payer_id = py.payer_id
                WHERE LOWER(TRIM(pd.state_name)) = LOWER(TRIM(%s))
                  AND LOWER(TRIM(py.payer_name)) = LOWER(TRIM(%s))
                  AND LOWER(TRIM(pd.plan_name)) ILIKE LOWER(TRIM(%s));
            """
            cursor.execute(exact_query, (state_name, payer, plan_name))
            result = cursor.fetchone()
            
            if result:
                plan_id, payer_id, db_payer_name, db_plan_name, formulary_url = result
                logger.info(f"Found exact match in DB: Plan='{db_plan_name}', Payer='{db_payer_name}'")
                return plan_id, payer_id, db_payer_name, db_plan_name, formulary_url

            logger.warning(f"No exact match for '{plan_name}'. Falling back to fuzzy matching...")
            cursor.execute("""
                SELECT pd.plan_id, pd.payer_id, py.payer_name, pd.plan_name, pd.state_name, pd.formulary_url
                FROM plan_details pd
                JOIN payer_details py ON pd.payer_id = py.payer_id
                WHERE LOWER(TRIM(pd.state_name)) = LOWER(TRIM(%s))
            """, (state_name,))
            
            all_records_in_state = cursor.fetchall()
            if not all_records_in_state:
                 logger.error(f"Fuzzy match failed: No plans found for state '{state_name}'")
                 return None, None, None, None, None

            best_match = None
            best_score = 0.65
            
            for record in all_records_in_state:
                plan_id, payer_id, db_payer_name, db_plan_name, db_state_name, formulary_url = record
                
                payer_score = similarity(normalized_payer, db_payer_name)
                plan_score = similarity(normalized_plan_name, db_plan_name)
                
                total_score = (payer_score * 0.3) + (plan_score * 0.7)
                
                if total_score > best_score:
                    best_score = total_score
                    best_match = record
            
            if best_match:
                plan_id, payer_id, db_payer_name, db_plan_name, _, formulary_url = best_match
                logger.info(f"Found fuzzy match (score: {best_score:.2f}): Plan='{db_plan_name}', Payer='{db_payer_name}'")
                return plan_id, payer_id, db_payer_name, db_plan_name, formulary_url

            logger.error(f"Fuzzy match failed for plan '{plan_name}' in state '{state_name}'. No suitable match found.")
            return None, None, None, None, None

        except Exception as e:
            logger.error(f"Error in get_plan_and_payer_info: {e}")
            return None, None, None, None, None

def process_pdfs_in_parallel():
    """Processes all PDFs in a folder in parallel using a ProcessPoolExecutor."""
    logger.info("STEP 2: Processing PDF Files in Parallel with ACCURATE LLM Strategy")

    all_processed_data = []
    local_raw_content = {} 

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"No PDF files found in '{PDF_FOLDER}'.")
        return [], {}

    logger.info(f"Found {len(pdf_files)} PDFs. Starting parallel processing with up to {PROCESS_COUNT} workers.")
    success_count, error_count, skipped_count = 0, 0, 0
    
    with ProcessPoolExecutor(max_workers=PROCESS_COUNT) as executor:
        futures = {executor.submit(process_single_pdf_worker, filename, PDF_FOLDER): filename for filename in pdf_files}
        
        for future in as_completed(futures):
            filename = futures[future]
            try:
                status, _, result_data, costs = future.result()

                if status == 'SUCCESS':
                    logger.info(f"Aggregating results for SUCCESSFUL file: {filename}")
                    success_count += 1
                    
                    payer_name = result_data['db_payer_name']
                    if costs['mistral_pages'] > 0:
                        track_mistral_cost(payer_name, costs['mistral_pages'])
                    if costs['bedrock_tokens'] > 0:
                        track_bedrock_cost_precalculated(
                            payer_name,
                            costs['bedrock_tokens'],
                            costs['bedrock_cost'],
                            costs['bedrock_calls']
                        )

                    all_processed_data.extend(result_data["processed_records"])
                
                elif status == 'SKIPPED':
                    logger.warning(f"Skipped file: {filename}. Reason: {result_data}")
                    skipped_count += 1
                
                elif status == 'ERROR':
                    logger.error(f"Error processing file: {filename}. Reason: {result_data}")
                    error_count += 1

            except Exception as e:
                logger.error(f"A critical error occurred while processing the result for {filename}: {e}", exc_info=True)
                error_count += 1

    logger.info("--- PDF Processing Complete ---")
    logger.info(f"Summary: {success_count} successful, {error_count} failed, {skipped_count} skipped")
    logger.info(f"Total structured records aggregated: {len(all_processed_data)}")
    return all_processed_data, local_raw_content


def process_single_pdf_worker(filename: str, pdf_folder_path: str):
    """Worker function using the new, highly accurate LLM pipeline with caching."""
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
        
        if structured_df is not None and not structured_df.empty:
            logger.info(f"{log_prefix} Cache HIT. Using pre-processed data for hash {file_hash[:10]}...")
            costs = zero_costs
        else:
            logger.info(f"{log_prefix} Cache MISS. Starting full processing for hash {file_hash[:10]}...")
            mistral_client = Mistral(api_key=MISTRAL_API_KEY)
            structured_df, raw_content, costs, all_acronyms, all_tiers = process_pdf_with_mistral_ocr(full_path, mistral_client, db_payer_name)
            
            dedup_acronyms = deduplicate_dicts(all_acronyms)
            dedup_tiers = deduplicate_dicts(all_tiers)

            if dedup_acronyms:
                insert_acronyms_to_ref_table(dedup_acronyms, state_name, payer, plan_name, "PP_Formulary_Short_Codes_Ref")
            if dedup_tiers:
                insert_acronyms_to_ref_table(dedup_tiers, state_name, payer, plan_name, "PP_Tier_Codes_Ref")

            if not structured_df.empty:
                cache_result(file_hash, structured_df, raw_content)

        if structured_df.empty:
            return 'SKIPPED', filename, "No structured data could be extracted by the LLM pipeline or cache.", costs
        
        processed_records = []
        for _, row in structured_df.iterrows():
            cleaned_drug_name = None
            coverage_status = None
            try:
                raw_drug_name = str(row.get('drug_name', '') or '')
                requirements_text = str(row.get('drug_requirements', '') or '').strip()

                cleaned_drug_name = clean_drug_name(raw_drug_name)

                if not cleaned_drug_name or len(cleaned_drug_name.strip()) < 2:
                    logger.debug(f"{log_prefix} Skipping short/empty drug name: {raw_drug_name!r}")
                    continue

                raw_tier = row.get('drug_tier', None)
                drug_tier_normalized = normalize_drug_tier(raw_tier)
                if not drug_tier_normalized:
                    drug_tier_normalized = infer_drug_tier_from_text(requirements_text) or infer_drug_tier_from_text(raw_drug_name)
                with get_db_connection() as conn:
                    coverage_status = determine_coverage_status(requirements_text, drug_tier_normalized, conn, state_name, db_payer_name)
                
                is_ql = "Yes" if "ql" in (requirements_text or "").lower() else "No"

                record = {
                    "id": str(uuid.uuid4()),
                    "plan_id": plan_id,
                    "payer_id": payer_id,
                    "drug_name": cleaned_drug_name,
                    "ndc_code": None,
                    "jcode": None,
                    "state_name": state_name,
                    "coverage_status": coverage_status,
                    "drug_tier": drug_tier_normalized,
                    "drug_requirements": requirements_text or None,
                    "is_prior_authorization_required": "Yes" if detect_prior_authorization(requirements_text) else "No",
                    "is_step_therapy_required": "Yes" if detect_step_therapy(requirements_text) else "No",
                    "is_quantity_limit_applied": is_ql,
                    "coverage_details": None,
                    "confidence_score": 0.95,
                    "source_url": formulary_url,
                    "plan_name": db_plan_name,
                    "payer_name": db_payer_name,
                    "file_name": filename
                }
                logger.debug(f"{log_prefix} Prepared record: name={record['drug_name']!r} tier={record['drug_tier']!r} coverage={record['coverage_status']!r}")
                processed_records.append(record)

            except Exception as e:
                logger.warning(f"{log_prefix} Error processing extracted row: {e}")
                continue
        
        if processed_records:
            logger.info(f"{log_prefix} Processed {len(processed_records)} drug records")
            result_payload = {
                "processed_records": processed_records,
                "db_payer_name": db_payer_name,
            }
            return 'SUCCESS', filename, result_payload, costs
        else:
            return 'SKIPPED', filename, "Data extracted, but no valid drug records could be processed from it.", costs
            
    except Exception as e:
        tb_str = traceback.format_exc()
        error_message = f"An unexpected error occurred in worker: {e}\n{tb_str}"
        logger.error(f"{log_prefix} ❌ {error_message}")
        return 'ERROR', filename, error_message, zero_costs

def get_all_plans_with_formulary_url():
    """Fetch all plans with a non-null formulary_url from the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pd.state_name, py.payer_name, pd.plan_name, pd.plan_id, py.payer_id, pd.formulary_url
            FROM plan_details pd
            JOIN payer_details py ON pd.payer_id = py.payer_id
            WHERE pd.formulary_url IS NOT NULL AND pd.formulary_url != ''
        """)
        return cursor.fetchall()

def process_single_pdf_url_worker(plan_info):
    """Worker: Download PDF from URL and process in-memory."""
    state_name, payer_name, plan_name, plan_id, payer_id, formulary_url = plan_info
    log_prefix = f"[Worker for {plan_name}]"
    zero_costs = {'mistral_pages': 0, 'bedrock_tokens': 0, 'bedrock_cost': 0.0, 'bedrock_calls': 0}
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(formulary_url, timeout=60, headers=headers)
        resp.raise_for_status()
        content_type = resp.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type:
            logger.error(f"{log_prefix} Invalid content type: {content_type}")
            return 'ERROR', plan_name, f"Invalid content type: {content_type}", zero_costs
        pdf_bytes = BytesIO(resp.content)

        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        structured_df, raw_content, costs, all_acronyms, all_tiers = process_pdf_with_mistral_ocr(pdf_bytes, mistral_client, payer_name)

        dedup_acronyms = deduplicate_dicts(all_acronyms)
        dedup_tiers = deduplicate_dicts(all_tiers)

        if dedup_acronyms:
            insert_acronyms_to_ref_table(dedup_acronyms, state_name, payer_name, plan_name, "PP_Formulary_Short_Codes_Ref")
        if dedup_tiers:
            insert_acronyms_to_ref_table(dedup_tiers, state_name, payer_name, plan_name, "PP_Tier_Codes_Ref")

        if structured_df.empty:
            return 'SKIPPED', plan_name, "No structured data extracted.", costs

        processed_records = []
        for _, row in structured_df.iterrows():
            cleaned_drug_name = clean_drug_name(str(row.get('drug_name', '') or ''))
            requirements_text = str(row.get('drug_requirements', '') or '').strip()
            if not cleaned_drug_name or len(cleaned_drug_name.strip()) < 2:
                continue
            
            drug_tier_normalized = normalize_drug_tier(row.get('drug_tier', None))
            if not drug_tier_normalized:
                drug_tier_normalized = infer_drug_tier_from_text(requirements_text) or infer_drug_tier_from_text(cleaned_drug_name)
                
            with get_db_connection() as conn:
                coverage_status = determine_coverage_status(requirements_text, drug_tier_normalized, conn, state_name, payer_name)
            
            is_pa = "Yes" if detect_prior_authorization(requirements_text) else "No"
            is_st = "Yes" if detect_step_therapy(requirements_text) else "No"
            is_ql = "Yes" if "ql" in (requirements_text or "").lower() else "No"

            record = {
                "id": str(uuid.uuid4()),
                "plan_id": plan_id,
                "payer_id": payer_id,
                "drug_name": cleaned_drug_name,
                "ndc_code": None,
                "jcode": None,
                "state_name": state_name,
                "coverage_status": coverage_status,
                "drug_tier": drug_tier_normalized,
                "drug_requirements": requirements_text or None,
                "is_prior_authorization_required": is_pa,
                "is_step_therapy_required": is_st,
                "is_quantity_limit_applied": is_ql,
                "coverage_details": None,
                "confidence_score": 0.95,
                "source_url": formulary_url,
                "plan_name": plan_name,
                "payer_name": payer_name,
                "file_name": f"{state_name}_{payer_name}_{plan_name}.pdf"
            }
            processed_records.append(record)
            
        if processed_records:
            result_payload = {
                "processed_records": processed_records,
                "db_payer_name": payer_name,
            }
            return 'SUCCESS', plan_name, result_payload, costs
        else:
            return 'SKIPPED', plan_name, "Data extracted, but no valid drug records.", costs
            
    except Exception as e:
        logger.error(f"{log_prefix} Error: {e}", exc_info=True)
        return 'ERROR', plan_name, str(e), zero_costs

def process_pdfs_from_urls_in_parallel():
    """Process PDFs by downloading from URLs in plan_details, in parallel, in-memory."""
    logger.info("STEP 2: Processing PDF Files from URLs in plan_details")
    all_processed_data = []

    plans = get_all_plans_with_formulary_url()
    if not plans:
        logger.warning("No plans with formulary URLs found.")
        return [], {}

    success_count, error_count, skipped_count = 0, 0, 0
    with ProcessPoolExecutor(max_workers=PROCESS_COUNT) as executor:
        futures = {executor.submit(process_single_pdf_url_worker, plan): plan for plan in plans}
        
        for future in as_completed(futures):
            plan_name_log = futures[future]
            
            try:
                status, _, result_data, costs = future.result()

                if status == 'SUCCESS':
                    logger.info(f"Aggregating results for SUCCESSFUL plan: {plan_name_log}")
                    success_count += 1
                    
                    payer_name = result_data['db_payer_name']
                    if costs['mistral_pages'] > 0:
                        track_mistral_cost(payer_name, costs['mistral_pages'])
                    if costs['bedrock_tokens'] > 0:
                        track_bedrock_cost_precalculated(
                            payer_name,
                            costs['bedrock_tokens'],
                            costs['bedrock_cost'],
                            costs['bedrock_calls']
                        )

                    all_processed_data.extend(result_data["processed_records"])
                
                elif status == 'SKIPPED':
                    logger.warning(f"Skipped plan: {plan_name_log}. Reason: {result_data}")
                    skipped_count += 1
                
                elif status == 'ERROR':
                    logger.error(f"Error processing plan: {plan_name_log}. Reason: {result_data}")
                    error_count += 1
            
            except Exception as e:
                logger.error(f"A critical error occurred while processing the result for {plan_name_log}: {e}", exc_info=True)
                error_count += 1

    logger.info("--- URL PDF Processing Complete ---")
    logger.info(f"Summary: {success_count} successful, {error_count} failed, {skipped_count} skipped")
    logger.info(f"Total structured records aggregated: {len(all_processed_data)}")
    return all_processed_data, {}


def deduplicate_dicts(dicts, keys=('acronym', 'expansion', 'explanation')):
    """
    Deduplicate a list of dicts based on the given keys, ignoring case and whitespace.
    This ensures we only remove rows that are functionally identical.
    """
    seen = set()
    deduped = []
    for d in dicts:
        key = tuple((d.get(k) or '').strip().lower() for k in keys)
        
        if key not in seen and any(val for val in key):
            seen.add(key)
            deduped.append(d)
    return deduped
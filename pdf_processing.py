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

from config import mistral_client, PDF_FOLDER, PROCESS_COUNT, MISTRAL_API_KEY, bedrock, MISTRAL_OCR_COST_PER_1K_PAGES, BEDROCK_COST_PER_1K_TOKENS, LLM_PAGE_WORKERS
from database import get_db_connection, get_cached_result, cache_result, update_plan_file_hash
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
    """
    logger.info("Extracting structured data from page markdown using Claude 3 Haiku...")
    
    costs = {'tokens': 0, 'cost': 0.0, 'calls': 1}

    system_prompt = """You are an expert data extraction agent... 
    Primary columns are "Drug name", "Drug tier", and "Requirements/Limits". [Drug Tier may be missing in some cases. Do not hallucinate there; just leave all values of drug_tier as null.]
    
    Few Variations [example]:
    drug_name: Match headers like: Drug Name, Medication, Brand Name, Generic Name, Formulary Drug, Product Name.
    drug_tier: Match headers like: Tier, Drug Tier, Brand or Generic, Formulary Tier, Cost Tier, Tier Level.
    drug_requirements: Match headers like: Requirements, Limits, Restrictions, Notes, Coverage Requirements and Limits, Requirements/Limits.
    
    Use keys: "drug_name", "drug_tier", "drug_requirements".

    CRITICAL:
    - Each value must be assigned to only one key: either "drug_name", "drug_tier", or "drug_requirements", based on its meaning and the column header. Do not duplicate any value across multiple keys.
    - IMPORTANT: Identify and Merge multiline drug names into a single drug_name.
    - If a row has a drug tier value in a separate column (e.g., "Tier 1", "TIER 2", "1", "T1"), populate "drug_tier" with exact values or null if unknown.
    - drug_requirements or drug tier may hold prior auth / QL / PA details.
    - Ignore section headers (ALL CAPS).
    - Value Mapping must be based on headers; One value belongs to one column only.
    - If a column header is split across lines (e.g., "Drug" on one line and "Tier" on the next), treat them as a single header (e.g., "Drug Tier") for correct value assignment.
    - [Drug Tier may be missing in some cases. Do not hallucinate there; just leave all values of drug_tier as null.]; Map as "drug_name": Drug Name, "drug_tier": Null, "drug_requirements": Requirements/ Limits
    
    INDEX PAGE DETECTION:
    - If you detect this is an INDEX or TABLE OF CONTENTS page (containing patterns like "Drug Name.......41", "Medication..........123", or lines with drug names followed by dots and page numbers), DO NOT extract any data.
    - INDEX INDICATORS: Lines with drug/medication names followed by multiple dots/periods and "ending with numbers" (page references).
    - If index content is detected, return an empty JSON array.

    Return ONLY a JSON array of objects with keys drug_name, drug_tier, drug_requirements.
    """

    user_message = f"""
    <EXAMPLE_INPUT>
    lipitor 10 mg ................ Tier 1
    PA; QL
    atorvastatin 20 mg ................ Tier 2
    PA
    </EXAMPLE_INPUT>

    <EXAMPLE_JSON>
    [
    {{
        "drug_name": "lipitor 10 mg",
        "drug_tier": "Tier 1",
        "drug_requirements": "PA; QL"
    }},
    {{
        "drug_name": "atorvastatin 20 mg",
        "drug_tier": "Tier 2",
        "drug_requirements": "PA"
    }}
    ]
    </EXAMPLE_JSON>

    ---
    Now process the following markdown:
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

        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'```json\s*(\[.*\])\s*```', response_text, re.DOTALL)
            if not json_match:
                logger.warning("LLM did not return a valid JSON array.")
                return [], costs
        
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
                return [], costs

        logger.info(f"Successfully extracted {len(structured_data)} records from page.")
        return structured_data, costs

    except Exception as e:
        logger.error(f"Error in Claude 3 Haiku LLM data extraction: {e}")
        traceback.print_exc()
        return [], costs


def process_pdf_with_mistral_ocr(pdf_path, mistral_client, payer_name=None):
    """
    New pipeline:
    1. Use Mistral OCR to get page-by-page markdown.
    2. Use a powerful LLM (Claude 3) to parse the markdown into structured JSON.
    3. Collect all page markdown for the `raw_content` cache.
    """
    logger.info(f"Analyzing PDF with parallel LLM pipeline: {os.path.basename(pdf_path)}")
    
    total_costs = {'mistral_pages': 0, 'mistral_cost': 0.0, 'bedrock_tokens': 0, 'bedrock_cost': 0.0, 'bedrock_calls': 0}
    
    try:
        pdf_file = Path(pdf_path)
        
        # 1. Upload and OCR with Mistral
        uploaded_file = mistral_client.files.upload(
            file={"file_name": pdf_file.stem, "content": pdf_file.read_bytes()},
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
                        all_structured_data.extend(structured_records)
                except Exception as exc:
                    logger.error(f"Page {page_num} generated an exception: {exc}")
        
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
         
        return structured_df, full_raw_content, total_costs

    except Exception as e:
        logger.error(f"Error in main PDF processing pipeline: {e}")
        traceback.print_exc()
        return pd.DataFrame(), "", total_costs

def get_plan_and_payer_info(state_name, payer, plan_name):
    """Get plan_id and payer_id from database with a more robust, query-based approach."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            logger.info(f"Looking for: State='{state_name}', Payer='{payer}', Plan='{plan_name}'")
            
            # normalized_payer = re.sub(r'[^a-zA-Z0-9 ]', '', str(payer)).replace(' ', '-').strip()
            # normalized_plan_name = re.sub(r'[^a-zA-Z0-9 ]', '', str(plan_name)).replace(' ', '-').strip()
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
            structured_df, raw_content, costs = process_pdf_with_mistral_ocr(full_path, mistral_client, db_payer_name)
            
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
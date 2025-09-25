import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import uuid

from config import ALL_PROCESSED_DATA, COST_TRACKER
from database import ensure_database_schema, insert_drug_formulary_data, update_plan_and_payer_statuses, update_drug_formulary_status
from excel_processing import populate_payer_and_plan_tables
from pdf_processing import process_pdfs_from_urls_in_parallel
from utils import validate_required_files, detect_step_therapy

logger = logging.getLogger(__name__)

def safe_create_directory(path):
    """Safely create directory with proper error handling""" 
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False

def save_cumulative_exports(all_processed_data):
    """Save all processed data to Excel and CSV files with enhanced error handling and new columns"""
    if not all_processed_data:
        logger.warning("No data to export")
        return
    
    full_df = pd.DataFrame(all_processed_data)
    
    required_columns = {
        "is_prior_authorization_required": "No",
        "is_step_therapy_required": "No", 
        "is_quantity_limit_applied": "No",   # <-- Add this line
        "status": "processing"
    }
    
    for col, default_value in required_columns.items():
        if col not in full_df.columns:
            logger.warning(f"Adding missing column '{col}' with default value '{default_value}'")
            full_df[col] = default_value
    
    # Map the "is_quantity_limit_applied" column to "Yes"/"No"
    full_df["is_quantity_limit_applied"] = full_df["is_quantity_limit_applied"].apply(
        lambda x: "Yes" if str(x).strip().lower() == "yes" else "No"
    )
    
    output_dir = Path("output_exports")
    if not safe_create_directory(output_dir):
        logger.error("Failed to create output directory")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        excel_path = output_dir / f"drug_formulary_complete_{timestamp}.xlsx"
        excel_df = full_df.copy()
        
        column_order = [
            "payer_name", "plan_name", "state_name", "drug_name", 
            "drug_tier", "drug_requirements", "coverage_status",
            "is_prior_authorization_required", "is_step_therapy_required",
            "is_quantity_limit_applied",   # <-- Add this line
            "status", "file_name", "id", "plan_id", "payer_id"
        ]
        
        for col in column_order:
            if col not in excel_df.columns:
                excel_df[col] = None
        
        excel_df = excel_df[column_order]
        
        excel_df.to_excel(excel_path, index=False)
        logger.info(f"Successfully exported data to Excel: {excel_path}")
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        
    try:
        csv_path = output_dir / f"drug_formulary_complete_{timestamp}.csv"
        full_df.to_csv(csv_path, index=False)
        logger.info(f"Successfully exported data to CSV: {csv_path}")
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")

def main():
    """Main function to run the entire data processing pipeline"""
    processed_plan_ids = []
    all_processed_data = []  # Initialize this variable to collect all processed data
    try:
        logger.info("========================================")
        logger.info("STARTING DRUG FORMULARY PROCESSING")
        logger.info("========================================")
        
        # Step 0: Initial Setup
        ensure_database_schema()
        validate_required_files()
        
        # Step 1: Populate Payer and Plan tables from Excel
        populate_payer_and_plan_tables()
        
        # Step 2: Process PDFs in parallel
        all_processed_data, _ = process_pdfs_from_urls_in_parallel()
        
        # Step 3: Insert data into the database
        if all_processed_data:
            logger.info("STEP 3: Inserting data into database")
            
            # Deduplicate data before insertion to avoid 'CardinalityViolation'
            logger.info(f"Deduplicating {len(all_processed_data)} records before insertion.")
            unique_records = {}
            for record in all_processed_data:
                # Key based on the ON CONFLICT constraint in the database
                key = (
                    record.get('plan_id'),
                    record.get('drug_name'),
                    record.get('drug_tier'),
                    record.get('drug_requirements')
                )
                if key not in unique_records:
                    unique_records[key] = record
            
            deduplicated_data = list(unique_records.values())
            records_removed = len(all_processed_data) - len(deduplicated_data)
            if records_removed > 0:
                logger.warning(f"Removed {records_removed} duplicate records.")
            
            logger.info("STEP 4: Saving Cumulative Data")
            save_cumulative_exports(deduplicated_data)

            
            logger.info(f"Proceeding with {len(deduplicated_data)} unique records.")
            insert_drug_formulary_data(deduplicated_data)
            
            # Collect unique plan IDs that were successfully processed
            processed_plan_ids = list(set(record['plan_id'] for record in deduplicated_data))
            
            # Update status to 'completed' for the inserted drugs
            update_drug_formulary_status(processed_plan_ids)
            
            # Update status to 'completed' for export
            for record in deduplicated_data:
                record['status'] = 'completed'
        else:
            logger.warning("Skipping database insertion and export as no data was processed.")

        # Step 5: Update final statuses in the database
        logger.info("STEP 5: Updating final plan and payer statuses.")
        update_plan_and_payer_statuses(processed_plan_ids)

        logger.info("========================================")
        logger.info("DRUG FORMULARY PROCESSING COMPLETE")
        logger.info("========================================")
        
    except Exception as e:
        logger.critical(f"A critical error occurred in the main pipeline: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cost report
        logger.info("\n--- FINAL COST & USAGE REPORT ---")
        for payer, costs in COST_TRACKER['payer_costs'].items():
            logger.info(f"\nPayer: {payer}")
            logger.info(f"  - PDFs Processed: {costs['pdfs_processed']}")
            logger.info(f"  - Mistral Pages: {costs['mistral_ocr_pages']}")
            logger.info(f"  - Mistral Cost: ${costs['mistral_cost']:.4f}")
            logger.info(f"  - Bedrock LLM Calls: {costs['llm_calls']}")
            logger.info(f"  - Bedrock Tokens: {costs['bedrock_tokens']}")
            logger.info(f"  - Bedrock Cost: ${costs['bedrock_cost']:.8f}")
            logger.info(f"  - Payer Total Cost: ${costs['total_cost']:.8f}")
        
        logger.info("\n--- OVERALL TOTALS ---")
        logger.info(f"Total PDFs Processed: {COST_TRACKER['total_pdfs_processed']}")
        logger.info(f"Total Mistral Pages: {COST_TRACKER['total_pages']}")
        logger.info(f"Total LLM Calls: {COST_TRACKER['total_llm_calls']}")
        logger.info(f"Total Bedrock Tokens: {COST_TRACKER['total_tokens']}")
        logger.info(f"GRAND TOTAL COST: ${COST_TRACKER['total_cost']:.8f}")

if __name__ == "__main__":
    main()

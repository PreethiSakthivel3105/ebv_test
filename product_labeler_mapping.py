import logging
import pandas as pd
import re
from psycopg2.extras import execute_batch
from database import get_db_connection

logger = logging.getLogger(__name__)

def map_product_labeler_codes():
    """
    Loads data using the existing DB connection, maps labeler codes to formulary drugs
    based on proprietary names, and performs an efficient bulk update.
    """
    logger.info("========================================")
    logger.info("STARTING PRODUCT LABELER MAPPING AND UPDATE PROCESS")
    logger.info("========================================")

    try:
        # Use the project's established context manager for all DB operations
        with get_db_connection() as conn:
            # --- 1. LOAD DATA ---
            logger.info("Loading required records from database...")
            
            # Fetch only records that haven't been mapped yet for maximum efficiency
            formulary_query = "SELECT id, drug_name FROM drug_formulary_details WHERE product_labeler_code IS NULL;"
            df_formulary = pd.read_sql(formulary_query, conn)
            
            if df_formulary.empty:
                logger.info("No new formulary records found to update. Skipping product mapping.")
                return

            logger.info(f"Loaded {len(df_formulary):,} formulary records needing a product match.")

            product_master_query = "SELECT product_labeler_code, proprietaryname FROM product_master;"
            df_product_master = pd.read_sql(product_master_query, conn)
            logger.info(f"Loaded {len(df_product_master):,} records from product_master.")

            # --- 2. PRE-PROCESS & MAP (in-memory with Pandas) ---
            logger.info("Pre-processing and mapping product data in memory...")
            df_product_master.dropna(subset=['proprietaryname'], inplace=True)
            df_product_master['proprietaryname_normalized'] = df_product_master['proprietaryname'].str.upper().str.strip()

            df_product_mapping = df_product_master.groupby('proprietaryname_normalized').agg(
                product_labeler_code=('product_labeler_code', 'first'),
                proprietaryname=('proprietaryname', 'first')
            ).reset_index()

            df_formulary['drug_name_normalized'] = df_formulary['drug_name'].str.upper()

            # Create a single, efficient regex pattern from all product names
            unique_names = sorted(df_product_mapping['proprietaryname_normalized'].unique(), key=len, reverse=True)
            escaped_names = [re.escape(name) for name in unique_names]
            pattern = '|'.join(escaped_names)

            # Extract matches in a single, vectorized operation
            df_formulary['matched_proprietary_name'] = df_formulary['drug_name_normalized'].str.extract(f'({pattern})')

            df_merged = pd.merge(
                df_formulary,
                df_product_mapping,
                left_on='matched_proprietary_name',
                right_on='proprietaryname_normalized',
                how='left'
            )

            df_to_update = df_merged.dropna(subset=['matched_proprietary_name']).copy()
            
            if df_to_update.empty:
                logger.info("No matches found between formulary and product master. Nothing to update.")
                return
            
            logger.info(f"Found {len(df_to_update):,} matches to update in the database.")

            # --- 3. EXECUTE BATCH UPDATE ---
            cursor = conn.cursor()
            update_query = """
                UPDATE drug_formulary_details
                SET product_labeler_code = %s, product_proprietaryname = %s
                WHERE id = %s;
            """
            
            # Prepare data as a list of tuples for the high-performance execute_batch function
            update_data = [
                (row['product_labeler_code'], row['proprietaryname'], row['id'])
                for index, row in df_to_update.iterrows()
            ]
            
            execute_batch(cursor, update_query, update_data, page_size=500)
            
            # The 'with get_db_connection()' context manager handles the commit automatically on success
            logger.info(f"✅ PRODUCT LABELER MAPPING AND UPDATE PROCESS COMPLETE. Updated {len(update_data):,} records.")

    except Exception as e:
        logger.error(f"A critical error occurred during the product mapping process: {e}", exc_info=True)

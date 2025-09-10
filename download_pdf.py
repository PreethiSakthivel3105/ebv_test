import pandas as pd
import requests
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re # For sanitizing filenames

# ——— Logging Setup ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ——— Config ———
CSV_PATH = Path("POC_AUG_Formularies_2025.csv")
DOWNLOAD_DIR = Path("druglist1")
MAX_WORKERS = 8

# ——— Helpers ———
def sanitize_filename(filename):
    """Sanitizes a string to be a valid filename and adds a .pdf extension."""
    s = str(filename).strip()
    s = re.sub(r'[\\/*?:"<>|]', "_", s) # Replace illegal characters
    s = s.replace(' ', '_') # Replace spaces for better compatibility
    if not s.lower().endswith('.pdf'):
        s += ".pdf"
    return s[:240] # Truncate to a safe length

def create_session():
    """Create requests session with retry logic"""
    session = requests.Session()
    retries = Retry(
        total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def download_pdf(url, filename, download_dir, session):
    """Download PDF from URL"""
    try:
        url = url.strip()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        logger.info(f"📥 Downloading: {filename}")
        response = session.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        filepath = download_dir / filename
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"✅ Downloaded: {filename}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to download {filename} from {url}. Reason: {e}")
        return False
    except OSError as e:
        logger.error(f"❌ OS Error saving {filename}. It may contain invalid characters. Error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred for {filename}: {e}")
        return False

def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed. Trying with 'cp1252' (common Windows encoding)...")
        try:
            df = pd.read_csv(CSV_PATH, encoding='cp1252')
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to read CSV with both 'utf-8' and 'cp1252' encodings. Error: {e}")
            return
    except FileNotFoundError:
        logger.error(f"❌ CRITICAL: The file '{CSV_PATH}' was not found.")
        return
    except Exception as e:
        logger.error(f"❌ CRITICAL: An unexpected error occurred while reading the CSV. Error: {e}")
        return

    try:
        df.columns = df.columns.str.strip().str.replace('\n', ' ')
        
        # Define the column names from your CSV file.
        URL_COLUMN = 'Formulary URL'
        COMPANY_COLUMN = 'Company Name' # This is your "Payer"
        PLAN_COLUMN = 'Plan Name'
        # <<< CHANGE 1: Add the column name for the state.
        STATE_COLUMN = 'States Covered'

        # Use the correct column names for data cleaning and validation.
        # <<< CHANGE 2: Add the state column to the list of required columns.
        required_cols = [URL_COLUMN, COMPANY_COLUMN, PLAN_COLUMN, STATE_COLUMN]
        df = df.dropna(subset=required_cols)
        for col in required_cols:
            df[col] = df[col].astype(str).str.strip()

    except KeyError as e:
        logger.error(f"❌ CRITICAL: The CSV is missing a required column: {e}. Please check the file headers.")
        return

    logger.info(f"📊 Found {len(df)} entries to process for download.")
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        session = create_session()
        futures = []

        for index, row in df.iterrows():
            # Get data from the correct columns.
            url = row[URL_COLUMN]
            company = row[COMPANY_COLUMN] # This is the "Payer"
            plan = row[PLAN_COLUMN]
            # <<< CHANGE 3: Extract the state from the row.
            state = row[STATE_COLUMN]

            # <<< CHANGE 4: Create the new filename in the format: State_Payer_Plan.pdf
            filename_raw = f"{state}_{company}_{plan}"

            if not url or not url.lower().startswith('http'):
                logger.warning(f"⏭️ Skipping row {index+2} due to missing or invalid URL.")
                continue

            # Sanitize the filename to prevent errors and add .pdf extension
            filename = sanitize_filename(filename_raw)

            filepath = DOWNLOAD_DIR / filename
            if filepath.exists() and filepath.stat().st_size > 0:
                logger.info(f"⏭️ Already exists: {filename}")
                success_count += 1
                continue
            
            futures.append(executor.submit(download_pdf, url, filename, DOWNLOAD_DIR, session))

        for future in as_completed(futures):
            if future.result():
                success_count += 1
            else:
                fail_count += 1

    logger.info("--- Download Complete ---")
    logger.info(f"📈 Summary: {success_count} successful (including existing), {fail_count} failed.")

if __name__ == "__main__":
    main()
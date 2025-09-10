import os
import pandas as pd
import requests
from pathlib import Path
from urllib.parse import urlparse
import re
import csv

# --- Config ---
EXCEL_PATH = r'POC_AUG.xlsx'
OUTPUT_DIR = Path("genentech_druglist_Updated_04_09")
START_ROW = 424  # 1-indexed row number

def sanitize_filename_part(text):
    """Sanitize individual parts of filename (state, company, plan) removing ONLY invalid characters"""
    # Convert to string and strip leading/trailing whitespace
    text = str(text).strip()
    # Remove ONLY invalid filename characters: / \ : * ? " < > |
    # Also remove newlines and tabs which could cause issues
    text = re.sub(r'[/\\:*?"<>|\r\n\t]', '', text)
    # Keep all other characters including spaces
    return text

def get_filename_from_url(url):
    """Extract filename from URL"""
    try:
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename or '.' not in filename:
            import urllib.parse as up
            query_params = up.parse_qs(parsed.query)
            for param in ['file', 'filename', 'document', 'doc']:
                if param in query_params:
                    filename = query_params[param][0]
                    break
        if not filename or '.' not in filename:
            filename = "formulary.pdf"
        return filename
    except:
        return "formulary.pdf"

def download_pdf(url, dest_path):
    """Download PDF from URL to destination path"""
    try:
        print(f"Downloading: {url}")
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Downloaded: {dest_path.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name="Formularies_AUG")
        print(f"Loaded {len(df)} rows from Excel.")
        df = df.iloc[START_ROW - 1:]
        print(f"Processing rows starting from row {START_ROW}...")
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return

    successful_downloads = 0
    failed_downloads = 0
    skipped_duplicates = 0
    filenames_seen = set()
    
    # Create CSV file to track failed downloads only
    failed_downloads_file = OUTPUT_DIR / "failed_downloads.csv"
    csv_headers = ['Excel_Row', 'State', 'Company', 'Plan_Name', 'URL', 'Error_Reason']
    
    with open(failed_downloads_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_headers)

        df_reset = df.reset_index()
        original_indices = df_reset['index'].values

        for idx, row in df_reset.iterrows():
            original_row_num = original_indices[idx] + 1
            print(f"\n--- Processing original row {original_row_num} (current: {idx + 1}/{len(df_reset)}) ---")

            url = str(row.get("Formulary URL", "")).strip()
            if not url or not url.startswith("http"):
                print(f"⚠️ Skipping row {original_row_num} — invalid URL: {url}")
                csv_writer.writerow([original_row_num, '', '', '', url, 'INVALID_URL'])
                failed_downloads += 1
                continue

            # Sanitize each part individually - preserving exact text from Excel
            state = sanitize_filename_part(row.get("States Covered", ""))
            if not state:
                state = "UnknownState"

            company = sanitize_filename_part(row.get("Company Name", ""))
            if not company:
                company = "UnknownCompany"

            plan_name = sanitize_filename_part(row.get("Plan Name", ""))
            if not plan_name:
                plan_name = "UnknownPlan"

            # Final filename: preserve exact formatting, separate parts with underscores only
            final_filename = f"{state}_{company}_{plan_name}.pdf"

            # Skip if duplicate filename detected
            if final_filename in filenames_seen:
                print(f"⚠️ Skipping duplicate filename: {final_filename}")
                csv_writer.writerow([original_row_num, state, company, plan_name, url, 'DUPLICATE_FILENAME'])
                skipped_duplicates += 1
                continue

            filenames_seen.add(final_filename)
            dest_path = OUTPUT_DIR / final_filename

            print(f"State: '{state}'")
            print(f"Company: '{company}'")
            print(f"Plan Name: '{plan_name}'")
            print(f"Final filename: {final_filename}")

            if download_pdf(url, dest_path):
                successful_downloads += 1
            else:
                csv_writer.writerow([original_row_num, state, company, plan_name, url, 'DOWNLOAD_FAILED'])
                failed_downloads += 1

    print(f"\n=== SUMMARY ===")
    print(f"Starting row: {START_ROW}")
    print(f"Rows processed: {len(df_reset)}")
    print(f"✅ Successful downloads: {successful_downloads}")
    print(f"❌ Failed downloads: {failed_downloads}")
    print(f"⏭️ Skipped duplicates: {skipped_duplicates}")
    print(f"📁 Files saved in: {OUTPUT_DIR.resolve()}")
    print(f"📝 Failed downloads logged in: {failed_downloads_file.resolve()}")

if __name__ == "__main__":
    main()
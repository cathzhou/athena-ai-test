# %%
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import ElementClickInterceptedException, ElementNotInteractableException, StaleElementReferenceException
from selenium.webdriver.support.ui import Select
import time
import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
# %%
os.system('rm -rf ~/.wdm')
# Initialize the browser
browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
browser.get("https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/compliance-actions-and-activities/warning-letters")
browser.refresh()
# %%
# Search for "dietary supplement"
search_box = browser.find_element(By.ID, "edit-search-api-fulltext")
search_box.send_keys("dietary supplement")
time.sleep(5)
dropdown = Select(browser.find_element(By.NAME, "datatable_length"))
dropdown.select_by_value("100")
# %%
data = []
for i in range(3):
    try:
        print("Parsing current page...")
        # Parse the table rows on the current page
        soup = BeautifulSoup(browser.page_source, "html.parser")
        table_rows = soup.select("table tbody tr")
        for row in table_rows:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue
            # Extract data as before
            letter_date = cols[0].text.strip()
            issue_date = cols[1].text.strip()
            link_tag = cols[2].find("a")
            company = link_tag.text.strip() if link_tag else ""
            href = link_tag["href"] if link_tag else ""
            full_url = "https://www.fda.gov" + href
            issuing_office = cols[3].text.strip()
            snippet = cols[-1].text.strip() if len(cols) >= 5 else ""

            data.append({
                "Letter Date": letter_date,
                "Issue Date": issue_date,
                "Company": company,
                "URL": full_url,
                "Issuing Office": issuing_office,
                "Snippet": snippet
            })
        # Wait for the "Next" button to become clickable
        wait = WebDriverWait(browser, 2)
        next_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#datatable_next a")))
        
        # Check if the "Next" button is disabled
        if "disabled" in next_button.get_attribute("class"):
            print("No more pages to load.")
            break
        
        # Scroll to the "Next" button and click
        next_button.click()
        time.sleep(10)  # Wait for the next page to load

    except Exception as e:
        print(f"Error navigating to the next page: {e}")
        break
# %%
df = pd.DataFrame(data)
# %%
os.chdir("/Users/catherinez/VSC/athena-ai/data")
# %%
# Save to CSV
df.to_csv("fda_dietary_supplement_warning_letters_all.csv", index=False)
# %%
def extract_letter_text(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, timeout=20)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the main content column
        main_content = soup.find('article', {'id': 'main-content'})

        # Find all paragraphs and list items inside it
        paragraphs = main_content.find_all(['p', 'li'])
        # Extract and clean text
        letter_text = '\n'.join(p.get_text(strip=True) for p in paragraphs)
        return letter_text
    except Exception as e:
        return f"Error: {e}"

print("Downloading full letter texts...")
for i, entry in enumerate(data):
    entry["Letter Text"] = extract_letter_text(entry["URL"])
    print(f"[{i+1}/{len(data)}] Fetched letter: {entry['Company']}")
    time.sleep(0.5)
# %%
df = pd.DataFrame(data)
# %%
df.to_csv("fda_dietary_supplement_warning_letters_with_text.csv", index=False)
# %%

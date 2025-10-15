import requests
from bs4 import BeautifulSoup

def get_stocks_from_moneycontrol(url):
    """
    Reads the provided Moneycontrol URL and extracts the 'Stocks to Watch'.
    It specifically skips other sections like earnings, deals, etc.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status() # This will check for any errors in accessing the page

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the main content area of the article
        article_content = soup.find('div', id='content_wrapper')
        if not article_content:
            print("--> Error: Could not find the main content block of the article.")
            return []

        all_paragraphs = article_content.find_all('p')
        news_items = []
        
        for p in all_paragraphs:
            headline_tag = p.find('strong')
            
            # We are looking for paragraphs that have a bolded headline
            if headline_tag:
                headline = headline_tag.get_text(strip=True).lower()
                
                # These are the sections we need to skip as per your request
                skip_keywords = [
                    'results today', 'quarterly earnings', 'bulk deal', 
                    'block deal', 'ipo corner', 'stocks in fn&o ban'
                ]

                # Check if the headline contains any of the keywords to skip
                if any(keyword in headline for keyword in skip_keywords):
                    continue # Go to the next paragraph

                # If it's a valid stock to watch, extract the full text
                full_text = p.get_text(strip=True)
                news_items.append(full_text)

        # The first item is usually an introduction, so we remove it
        if news_items and "stocks in focus" in news_items[0].lower():
            return news_items[1:]
        
        return news_items

    except requests.exceptions.RequestException as e:
        print(f"--> Network Error: Failed to retrieve the webpage. Please check your connection. Details: {e}")
        return None
    except Exception as e:
        print(f"--> An unexpected error occurred: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    target_url = "https://www.moneycontrol.com/news/business/markets/stocks-to-watch-today-rbl-bank-hcl-tech-lg-electronics-tata-motors-anand-rathi-wealth-kec-international-lodha-kfin-tech-in-focus-on-14-october-13613798.html"
    
    print(f"Reading website: {target_url}\n")
    
    stocks_to_watch = get_stocks_from_moneycontrol(target_url)
    
    if stocks_to_watch:
        print("--- Stocks to Watch ---")
        for i, item in enumerate(stocks_to_watch, 1):
            print(f"{i}. {item}\n")
    else:
        print("Could not extract the required information.")
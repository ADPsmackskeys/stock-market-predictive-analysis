
# Stocks-to-Watch Article Scraper

This Python script automates the extraction of all "Stocks to Watch" article URLs from Moneycontrol’s monthly sitemaps between January 2021 and September 2025.

## Overview

Moneycontrol publishes financial news articles, including "Stocks to Watch", in monthly sitemaps. This tool downloads those sitemaps and extracts URLs containing the keyword `"stocks-to-watch-today"` for further processing or analysis.

## Features

- Iterates from January 2021 to September 2025 over all monthly sitemaps.
- Parses sitemap XML using BeautifulSoup.
- Collects and deduplicates article URLs matching the pattern.
- Outputs the list of filtered URLs to the console.

## Usage

1. Clone the repository or download the script file.

2. Install required dependencies:

    ```bash
    pip install requests beautifulsoup4
    ```

3. Run the scraper:

    ```bash
    python scraper.py
    ```

4. The script will print the total number of articles found and list all URLs.

## Example Output

```
Total 'stocks-to-watch-today' articles found: 1250  
https://www.moneycontrol.com/news/business/markets/stocks-to-watch-today-august-31-2025-article-xyz.html  
https://www.moneycontrol.com/news/business/markets/stocks-to-watch-today-august-30-2025-article-abc.html  
...
```

## Customization

- To save the output to a CSV file, modify the script by adding:

    ```python
    import csv

    with open('stocks_to_watch_urls.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["URL"])
        for url in all_urls:
            writer.writerow([url])
    ```

- To change the date range or keyword, update the variables:

    ```python
    keyword = "your-keyword"
    years = range(start_year, end_year + 1)
    ```

## Notes

- Be mindful of the number of requests sent to avoid being rate-limited.
- Ensure compliance with Moneycontrol’s terms of service when scraping and using the data.

## License

This project is released under the MIT License.

import os
import pandas as pd

# Make a ticker file with only stocks with sufficient data
DATA_DIR = os.path.join ("data", "technical")
OUTPUT_FILE = os.path.join ("config", "valid_tickers_1000.py")
MIN_ROWS = 1000  # Ensures that there is less overfitting
valid_tickers = []
for file in os.listdir (DATA_DIR):
    if file.endswith (".csv"):
        filepath = os.path.join(DATA_DIR, file)
        try:
            df = pd.read_csv (filepath)
            if len (df) > MIN_ROWS:
                ticker = file.replace ("_data.csv", "")
                valid_tickers.append (ticker)
            else:
                print (f"{file} - Skipped")
        except Exception as e:
            print (f"Error reading {file} : {e}")
with open (OUTPUT_FILE, "w") as f:
    f.write("TICKERS = [\n")
    for ticker in valid_tickers:
        f.write (f'    "{ticker}", \n')
    f.write ("]\n")
print ("Saved Updated ticker list to {OUTPUT_FILE}")
import pandas as pd

# Load the Wikipedia page
url = "https://en.wikipedia.org/wiki/Nasdaq-100"
# Read all tables from the webpage
tables = pd.read_html(url)

# Assuming the correct table index is found after manual inspection
nasdaq_100_table_index = 4  # You need to replace '4' with the correct index after confirming
nasdaq_100_df = tables[nasdaq_100_table_index]

# Print the first few rows to verify it is the correct table
print(nasdaq_100_df.head())

# If 'Symbol' or 'Ticker' is the correct column name
tickers = nasdaq_100_df['Ticker'].tolist()  # Replace 'Symbol' with the correct column name if different
print(tickers)

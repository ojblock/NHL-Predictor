import pandas as pd

DATA_FILE = 'nhl_historical_stats.csv'
print(f"Loading data from {DATA_FILE}...")

try:
    df = pd.read_csv(DATA_FILE)
    initial_rows = len(df)
    print(f"Loaded {initial_rows} rows.")

    # Convert Date column if necessary (it should be loaded as object/string)
    # No need to convert to datetime for deduplication, keep as string for consistency
    
    # Define the columns that make a record unique (player within a specific game)
    unique_cols = ['Game_ID', 'Player_ID', 'Date']

    # Sort by date (important if multiple updates happen, ensure we keep the latest if time isn't tracked)
    # Keep the 'last' occurrence of duplicates based on the unique columns
    df_deduplicated = df.drop_duplicates(subset=unique_cols, keep='last')

    final_rows = len(df_deduplicated)
    rows_removed = initial_rows - final_rows

    if rows_removed > 0:
        print(f"Removed {rows_removed} duplicate player-game entries.")
        # Overwrite the original file with the cleaned data
        df_deduplicated.to_csv(DATA_FILE, index=False)
        print(f"Saved deduplicated data back to {DATA_FILE}.")
    else:
        print("No duplicate entries found.")

except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Cannot deduplicate.")
except Exception as e:
    print(f"An error occurred during deduplication: {e}")
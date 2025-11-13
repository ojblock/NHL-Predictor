import pandas as pd
import sys

print("--- Analyzing Player_ID column in nhl_featured_stats.csv ---")

try:
    df = pd.read_csv('nhl_featured_stats.csv')
    
    if 'Player_ID' not in df.columns:
        print("ERROR: 'Player_ID' column not found in the file!")
        sys.exit()

    # 1. Check for missing values
    missing_ids = df['Player_ID'].isna().sum()
    print(f"Total rows: {len(df)}")
    print(f"Rows with missing Player_ID: {missing_ids}")

    # 2. Print the data type (dtype)
    print(f"\nData type of 'Player_ID' column: {df['Player_ID'].dtype}")
    
    # 3. Print the first 5 non-missing Player_IDs
    print("\nFirst 5 Player_IDs in the file (as they are loaded):")
    print(df[df['Player_ID'].notna()]['Player_ID'].head())

    # 4. Try the integer conversion
    try:
        # Get all unique, non-missing Player_IDs
        unique_ids = df['Player_ID'].dropna().unique()
        
        # Try converting them
        converted_ids = [int(pid) for pid in unique_ids]
        print(f"\nSuccessfully converted {len(converted_ids)} unique Player_IDs to integer.")
        print(f"Sample of converted IDs: {converted_ids[:5]}")
        
    except Exception as e:
        print(f"\n---!!! FAILED to convert Player_IDs to integer !!!---")
        print(f"ERROR: {e}")
        print("This is the reason for the mismatch. The Player_ID column may contain non-numeric text or bad data.")

except FileNotFoundError:
    print("ERROR: nhl_featured_stats.csv not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
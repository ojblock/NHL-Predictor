import pandas as pd

def convert_time_to_seconds(time_str):
    if pd.isna(time_str) or not isinstance(time_str, str): return 0
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return (minutes * 60) + seconds
    except ValueError: return 0

print("Loading historical data...")
df = pd.read_csv('nhl_historical_stats.csv')
print("Preprocessing data...")
df['Date'] = pd.to_datetime(df['Date'])
for col in ['Time_On_Ice', 'PowerPlay_TOI', 'ShortHanded_TOI']:
    df[col] = df[col].apply(convert_time_to_seconds)
df['Did_Score'] = df['Goals'].apply(lambda x: 1 if x > 0 else 0)
df.to_csv('nhl_processed_stats.csv', index=False)
print("\n--- Preprocessing Complete ---")
print("Cleaned data saved to: nhl_processed_stats.csv")
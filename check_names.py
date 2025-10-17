import requests
import json
from datetime import datetime

print("--- Fetching today's schedule to check API structure ---")
try:
    # Let's find a date we know has games
    today_str = "2024-10-10" # Use a date from last year's season start
    schedule_url = f"https://api-web.nhle.com/v1/schedule/{today_str}"
    
    print(f"Fetching data for {today_str}...")
    schedule_data = requests.get(schedule_url).json()

    if not schedule_data.get('gameWeek'):
        print("No games found for this date to check.")
    else:
        # Get the very first game from that day
        first_game = schedule_data['gameWeek'][0]['games'][0]
        
        print("\n--- Raw Game Data Structure ---")
        # Pretty-print the JSON structure for just that one game
        print(json.dumps(first_game, indent=2))
        
        print("\n--- Analysis Complete ---")
        print("Please paste this entire output (especially the 'Raw Game Data Structure' block) back to me.")
        print("This will show me the exact location of the team names.")

except Exception as e:
    print(f"An error occurred: {e}")
import requests
import json
from datetime import datetime
import pytz

YOUR_TIMEZONE = 'America/Vancouver' 

print("--- Fetching tonight's schedule and ONE game roster ---")

try:
    local_tz = pytz.timezone(YOUR_TIMEZONE)
    now_local = datetime.now(local_tz)
    tonight_str = now_local.strftime('%Y-%m-%d')
    print(f"Fetching schedule for local date: {tonight_str}...")

    schedule_url = f"https://api-web.nhle.com/v1/schedule/{tonight_str}"
    schedule_data = requests.get(schedule_url).json()

    if not schedule_data.get('gameWeek') or not schedule_data['gameWeek'][0]['games']:
        print("No games found for tonight to check.")
    else:
        # Get the first game of the day
        first_game = schedule_data['gameWeek'][0]['games'][0]
        game_pk = first_game['id']
        
        print(f"Found game. Fetching roster for Game_ID: {game_pk}...")
        
        # Fetch the roster from the 'landing' endpoint
        landing_url = f"https://api-web.nhle.com/v1/gamecenter/{game_pk}/landing"
        game_data = requests.get(landing_url).json()
        
        roster_spots = game_data.get('rosterSpots', [])
        if not roster_spots:
            print("No roster spots found for this game.")
        else:
            print("\n--- Found Active Roster. Sample of 'playerId' values: ---")
            
            player_ids_from_api = []
            for player in roster_spots[:5]: # Get first 5 players
                player_ids_from_api.append(player.get('playerId'))
                
            print(json.dumps(player_ids_from_api, indent=2))
            
            print("\n--- Analysis ---")
            print("Please compare these IDs to the sample from your last check.")
            print("Are they integers? Are they strings? This will tell us why they aren't matching.")

except Exception as e:
    print(f"An error occurred: {e}")
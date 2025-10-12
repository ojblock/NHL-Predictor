import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def build_historical_dataset(start_date_str, end_date_str, session):
    """
    Fetches NHL player stats for all games within a given date range using a session object.
    """
    all_player_stats = []
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"Processing games for: {date_str}")
        
        try:
            schedule_url = f"https://statsapi.web.nhl.com/api/v1/schedule?date={date_str}"
            schedule_response = session.get(schedule_url)
            schedule_response.raise_for_status() # This will raise an error for bad responses (like 404, 500)
            schedule_data = schedule_response.json()
            
            if not schedule_data['dates']:
                current_date += timedelta(days=1)
                continue

            games = schedule_data['dates'][0]['games']
            
            for game in games:
                game_pk = game['gamePk']
                game_feed_url = f"https://statsapi.web.nhl.com/api/v1/game/{game_pk}/feed/live"
                
                # Add a small delay before fetching game data
                time.sleep(0.25) 
                
                game_response = session.get(game_feed_url)
                game_response.raise_for_status()
                game_data = game_response.json()

                boxscore = game_data['liveData']['boxscore']['teams']
                teams = [boxscore['away'], boxscore['home']]
                
                for team in teams:
                    for player_id, player_data in team['players'].items():
                        stats = player_data['stats'].get('skaterStats', {})
                        if stats:
                            player_info = {
                                'Game_ID': game_pk, 'Date': date_str, 'Player_ID': player_id.replace('ID', ''),
                                'Player_Name': player_data['person']['fullName'], 'Team': team['team']['name'],
                                'Goals': stats.get('goals', 0), 'Assists': stats.get('assists', 0),
                                'Shots': stats.get('shots', 0), 'Hits': stats.get('hits', 0),
                                'Blocked_Shots': stats.get('blocked', 0), 'Penalty_Minutes': stats.get('penaltyMinutes', 0),
                                'Time_On_Ice': stats.get('timeOnIce', '0:00'),
                                'PowerPlay_TOI': stats.get('powerPlayTimeOnIce', '0:00'),
                                'ShortHanded_TOI': stats.get('shortHandedTimeOnIce', '0:00')
                            }
                            all_player_stats.append(player_info)

        except requests.exceptions.RequestException as e:
            print(f"  - Could not process date {date_str}. Error: {e}")

        # Increased delay after processing all games for a single day
        time.sleep(1) 
        current_date += timedelta(days=1)
        
    return pd.DataFrame(all_player_stats)

# --- Main Execution ---
print("--- Starting Historical Data Collection ---")

session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)

# Define season ranges
season_23_24_start, season_23_24_end = "2023-10-10", "2024-04-18"
season_24_25_start, season_24_25_end = "2024-10-04", "2025-04-17"

# Build datasets
print("\n--- Building 2023-2024 Season Dataset ---")
df_23_24 = build_historical_dataset(season_23_24_start, season_23_24_end, session)

print("\n--- Building 2024-2025 Season Dataset ---")
df_24_25 = build_historical_dataset(season_24_25_start, season_24_25_end, session)

# Combine and Save
print("\n--- Combining Datasets ---")
master_df = pd.concat([df_23_24, df_24_25], ignore_index=True)

save_path = '/content/drive/MyDrive/nhl_historical_stats.csv'
master_df.to_csv(save_path, index=False)

print(f"\n--- COMPLETE ---")
if not master_df.empty:
    print(f"Historical dataset created with {len(master_df)} player-game records.")
    print(f"Data saved to: {save_path}")
else:
    print("No data was collected. Please check for persistent errors.")
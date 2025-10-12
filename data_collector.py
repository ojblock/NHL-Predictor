import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def build_historical_dataset(start_date_str, end_date_str, session):
    """
    Fetches NHL player stats using the NEW official NHL API.
    """
    all_player_stats = []
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"Processing games for: {date_str}")

        try:
            # 1. Use the new schedule endpoint
            schedule_url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
            schedule_response = session.get(schedule_url)
            schedule_response.raise_for_status()
            schedule_data = schedule_response.json()

            if not schedule_data.get('gameWeek'):
                current_date += timedelta(days=1)
                time.sleep(1)
                continue
            
            # 2. Loop through each game and get player stats from the new gamecenter endpoint
            for game in schedule_data['gameWeek'][0]['games']:
                game_pk = game['id']
                gamecenter_url = f"https://api-web.nhle.com/v1/gamecenter/{game_pk}/boxscore"
                
                time.sleep(0.25) # Small delay before fetching game data
                
                game_response = session.get(gamecenter_url)
                game_response.raise_for_status()
                game_data = game_response.json()

                # Process both away and home teams
                for team_key in ['awayTeam', 'homeTeam']:
                    # --- THIS IS THE CORRECTED LINE ---
                    # Use .get() for safe access in case the 'name' key is missing
                    team_name = game_data.get(team_key, {}).get('name', {}).get('default', 'Unknown Team')

                    # The new API has separate lists for player types
                    player_stats_section = game_data.get('playerByGameStats', {}).get(team_key, {})
                    for player in player_stats_section.get('forwards', []) + player_stats_section.get('defense', []):
                        player_info = {
                            'Game_ID': game_pk, 'Date': date_str, 'Player_ID': player['playerId'],
                            'Player_Name': player.get('name', {}).get('default', 'Unknown Player'), 'Team': team_name,
                            'Goals': player.get('goals', 0), 'Assists': player.get('assists', 0),
                            'Shots': player.get('shots', 0), 'Hits': player.get('hits', 0),
                            'Blocked_Shots': player.get('blockedShots', 0), 'Penalty_Minutes': player.get('pim', 0),
                            'Time_On_Ice': player.get('toi', '0:00'),
                            'PowerPlay_TOI': player.get('powerPlayToi', '0:00'),
                            'ShortHanded_TOI': player.get('shorthandedToi', '0:00')
                        }
                        all_player_stats.append(player_info)

        except requests.exceptions.RequestException as e:
            print(f"  - Could not process date {date_str}. Error: {e}")

        time.sleep(1) # Wait 1 second before processing the next day
        current_date += timedelta(days=1)
        
    return pd.DataFrame(all_player_stats)

# --- Main Execution ---
print("--- Starting Historical Data Collection (Using NEW API) ---")

session = requests.Session()
retry_strategy = Retry(
    total=5, backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)

# Define season ranges
season_23_24_start, season_23_24_end = "2023-10-10", "2024-04-18"
season_24_25_start, season_24_25_end = "2024-10-04", "2025-04-17" # Note: 24-25 is in the future

# Build datasets
print("\n--- Building 2023-2024 Season Dataset ---")
df_23_24 = build_historical_dataset(season_23_24_start, season_23_24_end, session)

print("\n--- Building 2024-2025 Season Dataset ---")
df_24_25 = build_historical_dataset(season_24_25_start, season_24_25_end, session)

# Combine and Save
print("\n--- Combining Datasets ---")
master_df = pd.concat([df_23_24, df_24_25], ignore_index=True)

save_path = 'nhl_historical_stats.csv'
master_df.to_csv(save_path, index=False)

print(f"\n--- COMPLETE ---")
if not master_df.empty:
    print(f"Historical dataset created with {len(master_df)} player-game records.")
    print(f"Data saved to: {save_path}")
else:
    print("No data was collected. Please check for persistent errors.")
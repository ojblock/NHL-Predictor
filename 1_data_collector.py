import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import sys

# --- Helper Function ---
def get_full_team_name(team_data):
    """Extracts and combines placeName and commonName."""
    try:
        place_name = team_data['placeName']['default']
        common_name = team_data['commonName']['default']
        # Handle potential None values safely
        if place_name and common_name:
            return f"{place_name} {common_name}"
        elif common_name: # Fallback if placeName is missing (e.g., All-Star games)
             return common_name
    except (KeyError, TypeError, AttributeError):
        pass # Catch potential errors if structure is unexpected
    return None # Return None if names cannot be constructed

def fetch_and_process_data_for_date(date_str, session):
    """Fetches and processes game data for a single specified date."""
    daily_player_stats = []
    print(f"Processing games for: {date_str}")
    try:
        schedule_url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
        schedule_data = session.get(schedule_url).json()

        if not schedule_data.get('gameWeek'):
            print(f"  - No games found on {date_str}.")
            return pd.DataFrame(daily_player_stats) # Return empty DataFrame

        for game in schedule_data['gameWeek'][0]['games']:
            home_team_name = get_full_team_name(game.get('homeTeam'))
            away_team_name = get_full_team_name(game.get('awayTeam'))

            if not home_team_name or not away_team_name:
                print(f"  - Skipping game {game.get('id', 'N/A')} due to missing team names.")
                continue

            game_pk = game['id']
            gamecenter_url = f"https://api-web.nhle.com/v1/gamecenter/{game_pk}/boxscore"
            time.sleep(0.25) # Short delay between API calls
            game_data = session.get(gamecenter_url).json()

            teams_data = { 'awayTeam': away_team_name, 'homeTeam': home_team_name }
            for team_key, team_name in teams_data.items():
                 # Check more robustly if player stats exist
                player_stats_section = game_data.get('playerByGameStats', {}).get(team_key)
                if player_stats_section:
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
                        daily_player_stats.append(player_info)
    except Exception as e:
        print(f"  - An error occurred processing {date_str}: {e}")

    return pd.DataFrame(daily_player_stats)

# --- Main Execution Logic ---
if __name__ == "__main__":
    # Determine the target date (yesterday)
    # Using UTC for GitHub Actions consistency, adjust if needed for specific timezone end-of-day
    yesterday_dt = datetime.utcnow() - timedelta(days=1)
    target_date_str = yesterday_dt.strftime('%Y-%m-%d')
    
    # Allow overriding date via command-line argument for testing/backfilling
    if len(sys.argv) > 1:
        try:
            # Validate the date format if an argument is passed
            datetime.strptime(sys.argv[1], '%Y-%m-%d')
            target_date_str = sys.argv[1]
            print(f"Using provided date: {target_date_str}")
        except ValueError:
            print(f"Invalid date format: {sys.argv[1]}. Using yesterday's date: {target_date_str}")

    DATA_FILE = 'nhl_historical_stats.csv'

    # Set up session for API requests
    session = requests.Session()
    retry_strategy = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    # Fetch data ONLY for the target date
    new_data_df = fetch_and_process_data_for_date(target_date_str, session)

    # Append data to the main file
    if not new_data_df.empty:
        print(f"\nAppending {len(new_data_df)} new records...")
        if os.path.exists(DATA_FILE):
            # Append without writing header if file exists
            new_data_df.to_csv(DATA_FILE, mode='a', header=False, index=False)
            print(f"Data for {target_date_str} successfully appended to {DATA_FILE}")
        else:
            # Create the file with header if it doesn't exist
            new_data_df.to_csv(DATA_FILE, mode='w', header=True, index=False)
            print(f"Created {DATA_FILE} with data for {target_date_str}")
    else:
        print(f"\nNo new data fetched for {target_date_str}. File not updated.")
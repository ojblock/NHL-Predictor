import pandas as pd
import joblib
import requests
from datetime import datetime
import pytz # Import the timezone library

# --- Configuration ---
PLAYER_LIST = [
    "C. McDavid", "A. Matthews", "N. MacKinnon", "D. Pastrnak",
    "L. Draisaitl", "M. Rantanen", "J. Robertson", "T. Thompson",
    "M. Tkachuk", "K. Kaprizov", "S. Crosby", "A. Ovechkin",
    "J. Hughes", "T. Stützle", "E. Pettersson"
]
MODEL_FILENAME = 'nhl_goal_predictor_model.joblib'
HISTORICAL_DATA_FILENAME = 'nhl_featured_stats.csv'
ROLL_WINDOW = 10
YOUR_TIMEZONE = 'America/Vancouver' # Use a Pacific Timezone

def get_full_team_name(team_data):
    try:
        place_name = team_data['placeName']['default']
        common_name = team_data['commonName']['default']
        return f"{place_name} {common_name}"
    except (KeyError, TypeError): return None

print("Loading model and historical data...")
try:
    model = joblib.load(MODEL_FILENAME)
    historical_df = pd.read_csv(HISTORICAL_DATA_FILENAME)
    historical_df['Date'] = pd.to_datetime(historical_df['Date'])
    print("Successfully loaded.")
except FileNotFoundError as e:
    print(f"Error: Could not find a necessary file. {e}"); exit()

print("\nFetching tonight's game schedules...")
# --- TIMEZONE CORRECTION ---
# Get the current time in your specified timezone
local_tz = pytz.timezone(YOUR_TIMEZONE)
now_local = datetime.now(local_tz)
tonight_str = now_local.strftime('%Y-%m-%d')
print(f"Fetching schedule for local date: {tonight_str}")
# --- END CORRECTION ---

schedule_url = f"https://api-web.nhle.com/v1/schedule/{tonight_str}"
schedule_data = requests.get(schedule_url).json()

if not schedule_data.get('gameWeek'):
    print("No games scheduled for tonight."); exit()

player_matchups = {}
for game in schedule_data['gameWeek'][0]['games']:
    home_team_name = get_full_team_name(game.get('homeTeam'))
    away_team_name = get_full_team_name(game.get('awayTeam'))
    if home_team_name and away_team_name:
        player_matchups[home_team_name] = away_team_name
        player_matchups[away_team_name] = home_team_name
print("Matchups fetched successfully.")

print("\nCalculating features for specified players...")
tonight_players_data = []
for player_name in PLAYER_LIST:
    player_all_stats = historical_df[historical_df['Player_Name'] == player_name]
    if player_all_stats.empty:
        print(f"  - Warning: No historical data found for {player_name}. Skipping.")
        continue
    
    player_recent_stats = player_all_stats.sort_values(by='Date').iloc[-1]
    player_team = player_recent_stats['Team']
    opponent_team = player_matchups.get(player_team)

    if not opponent_team:
        print(f"  - Info: {player_name}'s team ({player_team}) is not playing tonight ({tonight_str}). Skipping.")
        continue

    opponent_valid_stats = historical_df[historical_df['Team'] == opponent_team]
    if opponent_valid_stats.empty:
        print(f"  - Warning: No historical data found for opponent {opponent_team}. Skipping.")
        continue
    opponent_recent_stats = opponent_valid_stats.sort_values(by='Date').iloc[-1]

    # --- Feature Definition ---
    # Define the exact features your model expects IN THE CORRECT ORDER
    features_dict = {
        'Shots': player_recent_stats['Shots'], 
        'Hits': player_recent_stats['Hits'],
        'Blocked_Shots': player_recent_stats['Blocked_Shots'], 
        'Penalty_Minutes': player_recent_stats['Penalty_Minutes'],
        'Time_On_Ice': player_recent_stats['Time_On_Ice'], 
        'PowerPlay_TOI': player_recent_stats['PowerPlay_TOI'],
        'ShortHanded_TOI': player_recent_stats['ShortHanded_TOI'],
        'Avg_Goals_Last_10': player_recent_stats[f'Avg_Goals_Last_{ROLL_WINDOW}'],
        'Avg_Shots_Last_10': player_recent_stats[f'Avg_Shots_Last_{ROLL_WINDOW}'],
        'Avg_Time_On_Ice_Last_10': player_recent_stats[f'Avg_Time_On_Ice_Last_{ROLL_WINDOW}'],
        'Avg_PowerPlay_TOI_Last_10': player_recent_stats[f'Avg_PowerPlay_TOI_Last_{ROLL_WINDOW}'],
        'Avg_Hits_Last_10': player_recent_stats[f'Avg_Hits_Last_{ROLL_WINDOW}'],
        'Opp_GA_Avg_Last_10': opponent_recent_stats[f'Opp_GA_Avg_Last_{ROLL_WINDOW}']
    }
    # Add non-feature info for the final report
    features_dict['Player_Name'] = player_name
    features_dict['Team'] = player_team
    features_dict['Opponent'] = opponent_team
    tonight_players_data.append(features_dict)
    # --- End Feature Definition ---

tonight_df = pd.DataFrame(tonight_players_data)

if not tonight_df.empty:
    print("\nMaking predictions...")
    
    # --- Ensure correct feature order for prediction ---
    feature_order = model.get_booster().feature_names
    try:
        X_tonight = tonight_df[feature_order] 
    except KeyError as e:
        print(f"Error: Mismatch between features in data and model. Missing feature: {e}")
        print(f"Data columns: {tonight_df.columns.tolist()}")
        print(f"Model expected: {feature_order}")
        exit()
    # --- End Feature Order Check ---
        
    probabilities = model.predict_proba(X_tonight)[:, 1]
    tonight_df['Goal_Probability'] = probabilities
    results = tonight_df[['Player_Name', 'Team', 'Opponent', 'Goal_Probability']].sort_values(
        by='Goal_Probability', ascending=False
    )
    results['Goal_Probability'] = (results['Goal_Probability'] * 100).map('{:.2f}%'.format)
    print("\n--- Prediction Results ---")
    print(results.to_string(index=False))
else:
    print("\nCould not generate features for any of the specified players playing tonight.")
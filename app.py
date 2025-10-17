import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime
import pytz # For timezone

# --- Configuration (Match your predict_tonight.py) ---
MODEL_FILENAME = 'nhl_goal_predictor_model.joblib'
HISTORICAL_DATA_FILENAME = 'nhl_featured_stats.csv'
ROLL_WINDOW = 10
YOUR_TIMEZONE = 'America/Vancouver' # Use the same timezone

# --- Helper Functions (Copied from predict_tonight.py) ---
def get_full_team_name(team_data):
    """Extracts and combines placeName and commonName."""
    try:
        place_name = team_data['placeName']['default']
        common_name = team_data['commonName']['default']
        return f"{place_name} {common_name}"
    except (KeyError, TypeError):
        return None

# --- Load Model and Data (Do this once at the start) ---
@st.cache_resource # Cache the model so it doesn't reload every time
def load_model(filename):
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{filename}' not found. Please train the model first.")
        return None

@st.cache_data # Cache the data, refresh if the file changes
def load_data(filename):
    try:
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'])
        # Get unique player names for the dropdown, ensuring they are sorted
        player_names = sorted(df['Player_Name'].unique())
        return df, player_names
    except FileNotFoundError:
        st.error(f"Error: Data file '{filename}' not found. Please run the data pipeline.")
        return pd.DataFrame(), [] # Return empty objects

model = load_model(MODEL_FILENAME)
historical_df, unique_player_names = load_data(HISTORICAL_DATA_FILENAME)

# --- Load Raw Historical Data for Top Performers ---
# (Add this near where you load the other data file)
@st.cache_data
def load_raw_data(filename):
    try:
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error(f"Error: Raw data file '{filename}' not found.")
        return pd.DataFrame()

raw_historical_df = load_raw_data('nhl_historical_stats.csv') # Use the raw file

# --- Display Last Night's Top Performers ---
st.sidebar.header("Yesterday's Top Performers") # Add to sidebar for neatness

if not raw_historical_df.empty:
    try:
        # Get yesterday's date based on your timezone
        local_tz = pytz.timezone(YOUR_TIMEZONE)
        yesterday_dt = datetime.now(local_tz) - pd.Timedelta(days=1)
        yesterday_str = yesterday_dt.strftime('%Y-%m-%d')
        
        # Filter for yesterday's games
        yesterday_games = raw_historical_df[raw_historical_df['Date'] == yesterday_str].copy() # Use .copy() to avoid warnings

        if not yesterday_games.empty:
            # Calculate points
            yesterday_games['Points'] = yesterday_games['Goals'] + yesterday_games['Assists']
            
            # Aggregate stats per player
            player_stats = yesterday_games.groupby('Player_Name').agg(
                Team=('Team', 'first'), # Get the team name
                Goals=('Goals', 'sum'),
                Assists=('Assists', 'sum'),
                Points=('Points', 'sum')
            ).reset_index()
            
            # Sort by Points (descending), then Goals (descending)
            top_performers = player_stats.sort_values(by=['Points', 'Goals'], ascending=[False, False]).head(10) # Get top 10

            st.sidebar.dataframe(
                top_performers[['Player_Name', 'Team', 'Goals', 'Assists', 'Points']],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.sidebar.write(f"No game data found for {yesterday_str}.")
            
    except Exception as e:
        st.sidebar.error(f"Error processing top performers: {e}")
else:
    st.sidebar.warning("Raw historical data not loaded.")

# --- (Rest of your app code follows, like st.title, st.multiselect, etc.) ---
# --- Streamlit App Layout ---
st.set_page_config(layout="wide") # Use the full page width
st.title("🏒 NHL Goal Scorer Predictor")
st.write("Select players to predict their likelihood of scoring tonight.")

# --- UI: Player Selection ---
selected_players = st.multiselect(
    "Select Players:",
    options=unique_player_names,
)

# --- UI: Prediction Button ---
if st.button("Predict Probabilities"):
    if not selected_players:
        st.warning("Please select at least one player.")
    elif model is None or historical_df.empty:
        st.error("Model or data not loaded. Cannot make predictions.")
    else:
        # --- Backend Logic (Runs when button is clicked) ---
        with st.spinner("Fetching schedule and calculating predictions..."):
            
            # 1. Fetch Tonight's Matchups
            player_matchups = {}
            schedule_fetched_successfully = False # Flag
            try:
                local_tz = pytz.timezone(YOUR_TIMEZONE)
                now_local = datetime.now(local_tz)
                tonight_str = now_local.strftime('%Y-%m-%d')
                st.write(f"Fetching schedule for local date: {tonight_str}")

                schedule_url = f"https://api-web.nhle.com/v1/schedule/{tonight_str}"
                schedule_data = requests.get(schedule_url).json()

                if not schedule_data.get('gameWeek'):
                    st.error("No games scheduled for tonight.")
                else:
                    for game in schedule_data['gameWeek'][0]['games']:
                        home_team_name = get_full_team_name(game.get('homeTeam'))
                        away_team_name = get_full_team_name(game.get('awayTeam'))
                        if home_team_name and away_team_name:
                            player_matchups[home_team_name] = away_team_name
                            player_matchups[away_team_name] = home_team_name
                    # --- THIS LINE'S INDENTATION IS FIXED ---
                    schedule_fetched_successfully = True # Set flag on success
                    st.success("Matchups fetched successfully.")

            except Exception as e:
                st.error(f"Error fetching schedule: {e}")
                # Keep player_matchups as empty dict, flag remains False

            # 2. Engineer Features & Predict (only if schedule loaded)
            if schedule_fetched_successfully:
                tonight_players_data = []
                skipped_players = []

                for player_name in selected_players:
                    player_all_stats = historical_df[historical_df['Player_Name'] == player_name]
                    if player_all_stats.empty:
                        skipped_players.append(f"{player_name} (No historical data)")
                        continue
                    
                    player_recent_stats = player_all_stats.sort_values(by='Date').iloc[-1]
                    player_team = player_recent_stats['Team']
                    opponent_team = player_matchups.get(player_team)

                    if not opponent_team:
                        skipped_players.append(f"{player_name} (Team '{player_team}' not playing tonight)")
                        continue

                    opponent_valid_stats = historical_df[historical_df['Team'] == opponent_team]
                    if opponent_valid_stats.empty:
                         skipped_players.append(f"{player_name} (No data for opponent '{opponent_team}')")
                         continue
                    opponent_recent_stats = opponent_valid_stats.sort_values(by='Date').iloc[-1]

                    features_dict = {
                        'Shots': player_recent_stats['Shots'], 'Hits': player_recent_stats['Hits'],
                        'Blocked_Shots': player_recent_stats['Blocked_Shots'], 'Penalty_Minutes': player_recent_stats['Penalty_Minutes'],
                        'Time_On_Ice': player_recent_stats['Time_On_Ice'], 'PowerPlay_TOI': player_recent_stats['PowerPlay_TOI'],
                        'ShortHanded_TOI': player_recent_stats['ShortHanded_TOI'],
                        'Avg_Goals_Last_10': player_recent_stats.get(f'Avg_Goals_Last_{ROLL_WINDOW}', 0),
                        'Avg_Shots_Last_10': player_recent_stats.get(f'Avg_Shots_Last_{ROLL_WINDOW}', 0),
                        'Avg_Time_On_Ice_Last_10': player_recent_stats.get(f'Avg_Time_On_Ice_Last_{ROLL_WINDOW}', 0),
                        'Avg_PowerPlay_TOI_Last_10': player_recent_stats.get(f'Avg_PowerPlay_TOI_Last_{ROLL_WINDOW}', 0),
                        'Avg_Hits_Last_10': player_recent_stats.get(f'Avg_Hits_Last_{ROLL_WINDOW}', 0),
                        'Opp_GA_Avg_Last_10': opponent_recent_stats.get(f'Opp_GA_Avg_Last_{ROLL_WINDOW}', 0)
                    }
                    features_dict['Player_Name'] = player_name
                    features_dict['Team'] = player_team
                    features_dict['Opponent'] = opponent_team
                    tonight_players_data.append(features_dict)

                # 3. Display Results
                if tonight_players_data:
                    tonight_df = pd.DataFrame(tonight_players_data)
                    feature_order = model.get_booster().feature_names
                    X_tonight = tonight_df[feature_order]
                    probabilities = model.predict_proba(X_tonight)[:, 1]
                    tonight_df['Goal_Probability'] = probabilities
                    results = tonight_df[['Player_Name', 'Team', 'Opponent', 'Goal_Probability']].sort_values(
                        by='Goal_Probability', ascending=False
                    )
                    results_display = results.copy()
                    results_display['Goal_Probability'] = (results_display['Goal_Probability'] * 100).map('{:.2f}%'.format)
                    st.subheader("Prediction Results")
                    st.dataframe(results_display, use_container_width=True, hide_index=True)
                
                if skipped_players:
                    st.subheader("Skipped Players")
                    st.write("Could not generate predictions for the following players:")
                    for reason in skipped_players:
                        st.write(f"- {reason}")
                
                if not tonight_players_data and not skipped_players:
                     st.info("No predictions could be made for the selected players based on tonight's schedule.")

st.sidebar.info("App uses data up to the end of the previous day.")
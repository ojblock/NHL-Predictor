import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
import pytz # For timezone

# --- Configuration ---
MODEL_FILENAME = 'nhl_goal_predictor_model.joblib'
HISTORICAL_DATA_FILENAME = 'nhl_featured_stats.csv' # Data with features
RAW_DATA_FILENAME = 'nhl_historical_stats.csv'     # Raw daily data
ROLL_WINDOW = 10
YOUR_TIMEZONE = 'America/Vancouver' # Set to your local timezone

# --- Helper Functions ---
def get_full_team_name(team_data):
    """Extracts and combines placeName and commonName."""
    try:
        place_name = team_data['placeName']['default']
        common_name = team_data['commonName']['default']
        # Handle potential None values safely
        if place_name and common_name:
            return f"{place_name} {common_name}"
        elif common_name: # Fallback if placeName is missing
             return common_name
    except (KeyError, TypeError, AttributeError):
        pass
    return None

def convert_time_to_seconds(time_str):
    """Converts MM:SS to seconds (used for display if needed, main data is already processed)."""
    if pd.isna(time_str) or not isinstance(time_str, str): return 0
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return (minutes * 60) + seconds
    except ValueError: return 0

# --- Load Model and Data ---
@st.cache_resource # Cache the model resource
def load_model(filename):
    try:
        model = joblib.load(filename)
        # Store feature names used during training
        st.session_state['feature_order'] = model.get_booster().feature_names
        return model
    except FileNotFoundError:
        st.error(f"❌ Error: Model file '{filename}' not found. Please ensure '3_train_model.py' has run successfully and the model file is in the repository.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data # Cache the data, rerun if file changes
def load_data(filename):
    try:
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'])
        player_names = sorted(df['Player_Name'].unique())
        all_teams = sorted(df['Team'].unique())
        return df, player_names, all_teams
    except FileNotFoundError:
        st.error(f"❌ Error: Data file '{filename}' not found. Please ensure the data pipeline scripts (1, 2, 3) have run and files are in the repository.")
        return pd.DataFrame(), [], []
    except Exception as e:
        st.error(f"❌ Error loading data file '{filename}': {e}")
        return pd.DataFrame(), [], []

@st.cache_data
def load_raw_data(filename):
    try:
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error(f"❌ Error: Raw data file '{filename}' not found. Please ensure '1_data_collector.py' has run and the file is in the repository.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading raw data file '{filename}': {e}")
        return pd.DataFrame()

# --- Initialize ---
model = load_model(MODEL_FILENAME)
historical_df, unique_player_names, all_teams_in_data = load_data(HISTORICAL_DATA_FILENAME)
raw_historical_df = load_raw_data(RAW_DATA_FILENAME)
feature_order = st.session_state.get('feature_order', []) # Get feature order from session state

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("🏒 NHL Goal Scorer Predictor")

# --- Sidebar: Yesterday's Top Performers ---
st.sidebar.header("🗓️ Yesterday's Top Performers")
if not raw_historical_df.empty:
    try:
        local_tz = pytz.timezone(YOUR_TIMEZONE)
        yesterday_dt = datetime.now(local_tz) - timedelta(days=1)
        yesterday_str = yesterday_dt.strftime('%Y-%m-%d')

        # Filter using the datetime column directly for accuracy
        yesterday_games = raw_historical_df[raw_historical_df['Date'].dt.strftime('%Y-%m-%d') == yesterday_str].copy()

        if not yesterday_games.empty:
            yesterday_games['Points'] = yesterday_games['Goals'] + yesterday_games['Assists']
            player_stats = yesterday_games.groupby('Player_Name').agg(
                Team=('Team', 'first'),
                Goals=('Goals', 'sum'),
                Assists=('Assists', 'sum'),
                Points=('Points', 'sum')
            ).reset_index()
            top_performers = player_stats.sort_values(by=['Points', 'Goals'], ascending=[False, False]).head(10)
            st.sidebar.dataframe(
                top_performers[['Player_Name', 'Team', 'Goals', 'Assists', 'Points']],
                hide_index=True, use_container_width=True
            )
        else:
            st.sidebar.info(f"No game data found for yesterday ({yesterday_str}).")
    except Exception as e:
        st.sidebar.error(f"Error processing top performers: {e}")
else:
    st.sidebar.warning("Raw historical data not loaded.")
st.sidebar.info("App uses data up to the end of the previous day. Ensure GitHub Action ran.")

st.divider()

# --- Main Area 1: Top 5 Daily Predictions ---
st.header("🏆 Top 5 Predicted Scorers Tonight")

if st.button("Calculate Top 5 Predictions"):
    if model is None or historical_df.empty or not feature_order:
        st.error("Model or data not loaded correctly. Cannot make predictions.")
    else:
        with st.spinner("Fetching schedule, preparing data, and predicting... (This may take a moment)"):
            # 1. Fetch Tonight's Matchups & Teams Playing
            player_matchups = {}
            teams_playing_tonight = set()
            schedule_fetched_successfully = False
            try:
                local_tz = pytz.timezone(YOUR_TIMEZONE)
                now_local = datetime.now(local_tz)
                tonight_str = now_local.strftime('%Y-%m-%d')
                st.write(f"Fetching schedule for local date: {tonight_str}")
                schedule_url = f"https://api-web.nhle.com/v1/schedule/{tonight_str}"
                schedule_data = requests.get(schedule_url).json()

                if not schedule_data.get('gameWeek') or not schedule_data['gameWeek'][0]['games']:
                    st.info(f"No games scheduled for {tonight_str}.") # Use info instead of error
                else:
                    for game in schedule_data['gameWeek'][0]['games']:
                        home_team_name = get_full_team_name(game.get('homeTeam'))
                        away_team_name = get_full_team_name(game.get('awayTeam'))
                        if home_team_name and away_team_name:
                            player_matchups[home_team_name] = away_team_name
                            player_matchups[away_team_name] = home_team_name
                            teams_playing_tonight.add(home_team_name)
                            teams_playing_tonight.add(away_team_name)
                    schedule_fetched_successfully = True
                    st.success("Matchups fetched successfully.")

            except Exception as e:
                st.error(f"Error fetching schedule: {e}")

            # 2. Identify All Players Playing Tonight & Engineer Features
            if schedule_fetched_successfully and teams_playing_tonight:
                # Use idxmax() to get the index of the latest entry for each player
                latest_indices = historical_df.loc[historical_df.groupby('Player_ID')['Date'].idxmax()].index
                latest_player_stats = historical_df.loc[latest_indices]

                players_playing_tonight_df = latest_player_stats[latest_player_stats['Team'].isin(teams_playing_tonight)].copy()

                if players_playing_tonight_df.empty:
                    st.warning("Could not find recent stats for any players scheduled to play tonight.")
                else:
                    players_playing_tonight_df['Opponent'] = players_playing_tonight_df['Team'].map(player_matchups)
                    # Drop players whose opponent couldn't be mapped (e.g., team name mismatch)
                    players_playing_tonight_df.dropna(subset=['Opponent'], inplace=True)

                    if not players_playing_tonight_df.empty:
                        # Create opponent GA map efficiently
                        latest_team_def_stats = latest_player_stats.drop_duplicates(subset=['Team'], keep='last').set_index('Team')[f'Opp_GA_Avg_Last_{ROLL_WINDOW}']
                        players_playing_tonight_df[f'Opp_GA_Avg_Last_{ROLL_WINDOW}'] = players_playing_tonight_df['Opponent'].map(latest_team_def_stats).fillna(0)

                        # 3. Make Predictions
                        try:
                            X_tonight_all = players_playing_tonight_df[feature_order]
                            probabilities_all = model.predict_proba(X_tonight_all)[:, 1]
                            players_playing_tonight_df['Goal_Probability'] = probabilities_all

                            # 4. Get Top 5
                            top_5_results = players_playing_tonight_df[['Player_Name', 'Team', 'Opponent', 'Goal_Probability']].sort_values(
                                by='Goal_Probability', ascending=False
                            ).head(5)
                            top_5_display = top_5_results.copy()
                            top_5_display['Goal_Probability'] = (top_5_display['Goal_Probability'] * 100).map('{:.2f}%'.format)

                            # 5. Display Results
                            st.subheader("Top 5 Predicted Goal Scorers Tonight")
                            st.dataframe(top_5_display, use_container_width=True, hide_index=True)

                        except KeyError as e:
                            st.error(f"❌ Feature mismatch: Model expects feature '{e}' which is missing in the data. Check feature engineering steps.")
                            st.info(f"Available columns: {players_playing_tonight_df.columns.tolist()}")
                        except Exception as e:
                            st.error(f"❌ An error occurred during prediction: {e}")
                    else:
                        st.warning("Could not map opponents for players playing tonight.")
            elif schedule_fetched_successfully and not teams_playing_tonight:
                st.info("No games found in schedule data after processing team names.") # More specific message

st.divider()

# --- Main Area 2: Individual Player Prediction ---
st.header("🎯 Predict for Specific Players")
st.write("Select players below to predict their individual likelihood of scoring tonight.")

selected_players = st.multiselect(
    "Select Players:",
    options=unique_player_names,
    key="individual_player_select" # Unique key for this widget
)

if st.button("Predict for Selected Players"):
    if not selected_players:
        st.warning("Please select at least one player.")
    elif model is None or historical_df.empty or not feature_order:
        st.error("Model or data not loaded correctly. Cannot make predictions.")
    else:
        with st.spinner("Fetching schedule and calculating predictions..."):
            # --- Backend Logic (Runs when button is clicked) ---
            # (This logic is very similar to the Top 5 section but filtered)
            
            # 1. Fetch Matchups (Could potentially reuse from above if run on same page load)
            player_matchups = {}
            schedule_fetched_successfully = False
            try:
                local_tz = pytz.timezone(YOUR_TIMEZONE)
                now_local = datetime.now(local_tz)
                tonight_str = now_local.strftime('%Y-%m-%d')
                st.write(f"Fetching schedule for local date: {tonight_str}")
                schedule_url = f"https://api-web.nhle.com/v1/schedule/{tonight_str}"
                schedule_data = requests.get(schedule_url).json()

                if not schedule_data.get('gameWeek') or not schedule_data['gameWeek'][0]['games']:
                     st.info(f"No games scheduled for {tonight_str}.")
                else:
                    for game in schedule_data['gameWeek'][0]['games']:
                        home_team_name = get_full_team_name(game.get('homeTeam'))
                        away_team_name = get_full_team_name(game.get('awayTeam'))
                        if home_team_name and away_team_name:
                            player_matchups[home_team_name] = away_team_name
                            player_matchups[away_team_name] = home_team_name
                    schedule_fetched_successfully = True
                    st.success("Matchups fetched successfully.")
            except Exception as e:
                st.error(f"Error fetching schedule: {e}")

            # 2. Engineer Features & Predict for Selected Players
            if schedule_fetched_successfully:
                tonight_players_data = []
                skipped_players = []

                # Get latest team defensive stats map (needed for opponent GA)
                latest_indices = historical_df.loc[historical_df.groupby('Player_ID')['Date'].idxmax()].index
                latest_player_stats_all = historical_df.loc[latest_indices]
                latest_team_def_stats = latest_player_stats_all.drop_duplicates(subset=['Team'], keep='last').set_index('Team')[f'Opp_GA_Avg_Last_{ROLL_WINDOW}']


                for player_name in selected_players:
                    player_all_stats = historical_df[historical_df['Player_Name'] == player_name]
                    if player_all_stats.empty:
                        skipped_players.append(f"{player_name} (No historical data)"); continue
                    
                    player_recent_stats = player_all_stats.sort_values(by='Date').iloc[-1]
                    player_team = player_recent_stats['Team']
                    opponent_team = player_matchups.get(player_team)

                    if not opponent_team:
                        skipped_players.append(f"{player_name} (Team '{player_team}' not playing tonight)"); continue

                    # Use the pre-calculated map for opponent GA
                    opponent_ga_avg = latest_team_def_stats.get(opponent_team, 0) # Default to 0 if opponent not found

                    # Build features
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
                        'Opp_GA_Avg_Last_10': opponent_ga_avg # Use mapped value
                    }
                    # Add non-feature info
                    features_dict['Player_Name'] = player_name
                    features_dict['Team'] = player_team
                    features_dict['Opponent'] = opponent_team
                    tonight_players_data.append(features_dict)

                # 3. Display Results
                if tonight_players_data:
                    tonight_df = pd.DataFrame(tonight_players_data)
                    try:
                        X_tonight_selected = tonight_df[feature_order]
                        probabilities_selected = model.predict_proba(X_tonight_selected)[:, 1]
                        tonight_df['Goal_Probability'] = probabilities_selected
                        results = tonight_df[['Player_Name', 'Team', 'Opponent', 'Goal_Probability']].sort_values(
                            by='Goal_Probability', ascending=False
                        )
                        results_display = results.copy()
                        results_display['Goal_Probability'] = (results_display['Goal_Probability'] * 100).map('{:.2f}%'.format)
                        st.subheader("Prediction Results for Selected Players")
                        st.dataframe(results_display, use_container_width=True, hide_index=True)
                    except KeyError as e:
                         st.error(f"❌ Feature mismatch: Model expects feature '{e}' which is missing. Check feature engineering.")
                         st.info(f"Available columns: {tonight_df.columns.tolist()}")
                    except Exception as e:
                         st.error(f"❌ An error occurred during prediction: {e}")

                if skipped_players:
                    st.subheader("Skipped Players")
                    st.write("Could not generate predictions for the following selected players:")
                    for reason in skipped_players: st.write(f"- {reason}")
                
                if not tonight_players_data and not skipped_players:
                     st.info("No predictions could be made for the selected players based on tonight's schedule.")
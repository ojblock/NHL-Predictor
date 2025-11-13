import streamlit as st
import pandas as pd
import joblib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
import pytz 

# --- Configuration ---
MODEL_FILENAME = 'nhl_goal_predictor_model.joblib'
HISTORICAL_DATA_FILENAME = 'nhl_featured_stats.csv' 
RAW_DATA_FILENAME = 'nhl_historical_stats.csv'     
ROLL_WINDOW = 10
YOUR_TIMEZONE = 'America/Vancouver' 

# --- Helper Functions ---
def get_full_team_name(team_data):
    """Extracts and combines placeName and commonName."""
    try:
        place_name = team_data['placeName']['default']
        common_name = team_data['commonName']['default']
        if place_name and common_name:
            return f"{place_name} {common_name}"
        elif common_name:
             return common_name
    except (KeyError, TypeError, AttributeError):
        pass
    return None

@st.cache_resource(ttl=3600) 
def get_requests_session():
    """Creates a robust requests session with retries."""
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    return session

# --- 1. CHANGE HERE: Removed 'session' from the arguments ---
@st.cache_data(ttl=600) 
def get_tonights_schedule_and_rosters(date_str):
    """
    Fetches tonight's schedule AND all active player IDs from the gamecenter landing endpoints.
    """
    
    # --- 2. CHANGE HERE: Get the session *inside* the function ---
    session = get_requests_session() 
    
    st.write(f"Fetching schedule and active rosters for local date: {date_str}...")
    player_matchups = {}
    teams_playing_tonight = set()
    active_roster_player_ids = set() 

    # 1. Fetch Schedule
    schedule_url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
    schedule_data = session.get(schedule_url).json()

    if not schedule_data.get('gameWeek') or not schedule_data['gameWeek'][0]['games']:
        st.info(f"No games scheduled for {date_str}.")
        return None, None, None 

    # 2. Loop through games to get matchups AND rosters
    game_pks = []
    for game in schedule_data['gameWeek'][0]['games']:
        game_pks.append(game['id'])
        home_team_name = get_full_team_name(game.get('homeTeam'))
        away_team_name = get_full_team_name(game.get('awayTeam'))
        if home_team_name and away_team_name:
            player_matchups[home_team_name] = away_team_name
            player_matchups[away_team_name] = home_team_name
            teams_playing_tonight.add(home_team_name)
            teams_playing_tonight.add(away_team_name)

    # 3. Fetch Active Rosters for each game
    for game_pk in game_pks:
        try:
            landing_url = f"https://api-web.nhle.com/v1/gamecenter/{game_pk}/landing"
            game_data = session.get(landing_url).json()
            
            for player in game_data.get('rosterSpots', []):
                active_roster_player_ids.add(player['playerId'])
        except Exception as e:
            st.warning(f"Could not fetch roster for game {game_pk}: {e}")
            
    st.success("Matchups and active rosters fetched successfully.")
    return player_matchups, teams_playing_tonight, active_roster_player_ids

# --- Load Model and Data ---
@st.cache_resource
def load_model(filename):
    try:
        model = joblib.load(filename)
        st.session_state['feature_order'] = model.get_booster().feature_names
        return model
    except FileNotFoundError:
        st.error(f"❌ Error: Model file '{filename}' not found. Please ensure '3_train_model.py' has run successfully and the model file is in the repository.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_data(filename):
    try:
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'])
        player_names = sorted(df['Player_Name'].unique())
        return df, player_names
    except FileNotFoundError:
        st.error(f"❌ Error: Data file '{filename}' not found. Please ensure the data pipeline scripts (1, 2, 3) have run and files are in the repository.")
        return pd.DataFrame(), []
    except Exception as e:
        st.error(f"❌ Error loading data file '{filename}': {e}")
        return pd.DataFrame(), []

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
# NO LONGER NEED TO CREATE SESSION HERE
model = load_model(MODEL_FILENAME)
historical_df, unique_player_names = load_data(HISTORICAL_DATA_FILENAME)
raw_historical_df = load_raw_data(RAW_DATA_FILENAME)
feature_order = st.session_state.get('feature_order', [])

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

        yesterday_games = raw_historical_df[raw_historical_df['Date'].dt.strftime('%Y-%m-%d') == yesterday_str].copy()

        if not yesterday_games.empty:
            yesterday_games['Points'] = yesterday_games['Goals'] + yesterday_games['Assists']
            player_stats = yesterday_games.groupby('Player_Name').agg(
                Team=('Team', 'first'), Goals=('Goals', 'sum'),
                Assists=('Assists', 'sum'), Points=('Points', 'sum')
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
        with st.spinner("Fetching schedule, checking active rosters, and predicting..."):
            
            local_tz = pytz.timezone(YOUR_TIMEZONE)
            now_local = datetime.now(local_tz)
            tonight_str = now_local.strftime('%Y-%m-%d')
            
            # --- 3. CHANGE HERE: Removed 'session' from the call ---
            player_matchups, teams_playing_tonight, active_roster_ids = get_tonights_schedule_and_rosters(tonight_str)

            if schedule_fetched_successfully := (player_matchups is not None):
                
                latest_indices = historical_df.loc[historical_df.groupby('Player_ID')['Date'].idxmax()].index
                latest_player_stats = historical_df.loc[latest_indices]

                players_playing_tonight_df = latest_player_stats[latest_player_stats['Player_ID'].isin(active_roster_ids)].copy()

                if players_playing_tonight_df.empty:
                    st.warning("Could not find recent stats for any players on tonight's active rosters.")
                else:
                    players_playing_tonight_df['Opponent'] = players_playing_tonight_df['Team'].map(player_matchups)
                    players_playing_tonight_df.dropna(subset=['Opponent'], inplace=True) 

                    if not players_playing_tonight_df.empty:
                        latest_team_def_stats = latest_player_stats.drop_duplicates(subset=['Team'], keep='last').set_index('Team')[f'Opp_GA_Avg_Last_{ROLL_WINDOW}']
                        players_playing_tonight_df[f'Opp_GA_Avg_Last_{ROLL_WINDOW}'] = players_playing_tonight_df['Opponent'].map(latest_team_def_stats).fillna(0)

                        try:
                            X_tonight_all = players_playing_tonight_df[feature_order]
                            probabilities_all = model.predict_proba(X_tonight_all)[:, 1]
                            players_playing_tonight_df['Goal_Probability'] = probabilities_all

                            top_5_results = players_playing_tonight_df[['Player_Name', 'Team', 'Opponent', 'Goal_Probability']].sort_values(
                                by='Goal_Probability', ascending=False
                            ).head(5)
                            top_5_display = top_5_results.copy()
                            top_5_display['Goal_Probability'] = (top_5_display['Goal_Probability'] * 100).map('{:.2f}%'.format)

                            st.subheader("Top 5 Predicted Goal Scorers Tonight")
                            st.dataframe(top_5_display, use_container_width=True, hide_index=True)

                        except KeyError as e:
                            st.error(f"❌ Feature mismatch: Model expects feature '{e}'.")
                        except Exception as e:
                            st.error(f"❌ An error occurred during prediction: {e}")
                    else:
                        st.warning("Could not map opponents for players playing tonight.")

st.divider()

# --- Main Area 2: Individual Player Prediction ---
st.header("🎯 Predict for Specific Players")
st.write("Select players below to predict their individual likelihood of scoring tonight.")

selected_players = st.multiselect(
    "Select Players:",
    options=unique_player_names,
    key="individual_player_select"
)

if st.button("Predict for Selected Players"):
    if not selected_players:
        st.warning("Please select at least one player.")
    elif model is None or historical_df.empty or not feature_order:
        st.error("Model or data not loaded correctly. Cannot make predictions.")
    else:
        with st.spinner("Fetching schedule, checking active rosters, and predicting..."):
            
            local_tz = pytz.timezone(YOUR_TIMEZONE)
            now_local = datetime.now(local_tz)
            tonight_str = now_local.strftime('%Y-%m-%d')
            
            # --- 3. CHANGE HERE (Again): Removed 'session' from the call ---
            player_matchups, teams_playing_tonight, active_roster_ids = get_tonights_schedule_and_rosters(tonight_str)

            if schedule_fetched_successfully := (player_matchups is not None):
                tonight_players_data = []
                skipped_players = []

                latest_indices = historical_df.loc[historical_df.groupby('Player_ID')['Date'].idxmax()].index
                latest_player_stats_all = historical_df.loc[latest_indices]
                latest_team_def_stats = latest_player_stats_all.drop_duplicates(subset=['Team'], keep='last').set_index('Team')[f'Opp_GA_Avg_Last_{ROLL_WINDOW}']

                for player_name in selected_players:
                    player_all_stats = historical_df[historical_df['Player_Name'] == player_name]
                    if player_all_stats.empty:
                        skipped_players.append(f"{player_name} (No historical data)"); continue
                    
                    player_recent_stats = player_all_stats.sort_values(by='Date').iloc[-1]
                    player_team = player_recent_stats['Team']
                    
                    player_id = player_recent_stats.get('Player_ID')
                    if not player_id or player_id not in active_roster_ids:
                        skipped_players.append(f"{player_name} (Not on tonight's active roster)")
                        continue

                    opponent_team = player_matchups.get(player_team)
                    if not opponent_team:
                        skipped_players.append(f"{player_name} (Team '{player_team}' not playing tonight)"); continue

                    opponent_ga_avg = latest_team_def_stats.get(opponent_team, 0) 

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
                        'Opp_GA_Avg_Last_10': opponent_ga_avg
                    }
                    features_dict['Player_Name'] = player_name
                    features_dict['Team'] = player_team
                    features_dict['Opponent'] = opponent_team
                    tonight_players_data.append(features_dict)

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
                         st.error(f"❌ Feature mismatch: Model expects feature '{e}'.")
                    except Exception as e:
                         st.error(f"❌ An error occurred during prediction: {e}")

                if skipped_players:
                    st.subheader("Skipped Players")
                    st.write("Could not generate predictions for the following selected players:")
                    for reason in skipped_players: st.write(f"- {reason}")
                
                if not tonight_players_data and not skipped_players:
                     st.info("No predictions could be made for the selected players based on tonight's schedule.")

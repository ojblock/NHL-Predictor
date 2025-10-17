import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

print("Loading preprocessed data...")
df = pd.read_csv('nhl_processed_stats.csv')
df['Date'] = pd.to_datetime(df['Date'])
print("Engineering player performance features...")
df = df.sort_values(by=['Player_ID', 'Date'])
stats_to_roll = ['Goals', 'Shots', 'Time_On_Ice', 'PowerPlay_TOI', 'Hits']
roll_window = 10
for stat in stats_to_roll:
    df[f'Avg_{stat}_Last_{roll_window}'] = df.groupby('Player_ID')[stat].transform(
        lambda x: x.rolling(window=roll_window, min_periods=1).mean().shift(1)
    )
print("Engineering opponent strength features...")
team_goals = df.groupby(['Game_ID', 'Team', 'Date'])['Goals'].sum().reset_index()
merged_goals = pd.merge(team_goals, team_goals, on='Game_ID', suffixes=('_team', '_opp'))
goals_allowed = merged_goals[merged_goals['Team_team'] != merged_goals['Team_opp']]
team_defensive_stats = goals_allowed[['Date_team', 'Game_ID', 'Team_team', 'Goals_opp']].rename(
    columns={'Date_team': 'Date', 'Team_team': 'Team', 'Goals_opp': 'Goals_Allowed'}
)
team_defensive_stats = team_defensive_stats.sort_values(by=['Team', 'Date'])
team_defensive_stats[f'Opp_GA_Avg_Last_{roll_window}'] = team_defensive_stats.groupby('Team')['Goals_Allowed'].transform(
    lambda x: x.rolling(window=roll_window, min_periods=1).mean().shift(1)
)
game_teams = df[['Game_ID', 'Team']].drop_duplicates()
game_teams_merged = pd.merge(game_teams, game_teams, on='Game_ID', suffixes=('_A', '_B'))
opponent_map = game_teams_merged[game_teams_merged['Team_A'] != game_teams_merged['Team_B']]
opponent_map = opponent_map[['Game_ID', 'Team_A', 'Team_B']].rename(columns={'Team_A': 'Team', 'Team_B': 'Opponent'})
print("Merging all features together...")
df = pd.merge(df, opponent_map, on=['Game_ID', 'Team'], how='left')
df = pd.merge(df, team_defensive_stats[['Game_ID', 'Team', f'Opp_GA_Avg_Last_{roll_window}']],
              left_on=['Game_ID', 'Opponent'], right_on=['Game_ID', 'Team'],
              how='left', suffixes=('', '_drop'))
df = df.drop(columns=[col for col in df.columns if 'drop' in col or col == 'Team_y'])
df.rename(columns={'Team_x': 'Team'}, inplace=True)
df.fillna(0, inplace=True)
df.to_csv('nhl_featured_stats.csv', index=False)
print("Feature engineering complete. File saved.")
print("\nDefining features and target...")
target = 'Did_Score'
features = [
    'Shots', 'Hits', 'Blocked_Shots', 'Penalty_Minutes', 'Time_On_Ice', 
    'PowerPlay_TOI', 'ShortHanded_TOI', 'Avg_Goals_Last_10', 'Avg_Shots_Last_10', 
    'Avg_Time_On_Ice_Last_10', 'Avg_PowerPlay_TOI_Last_10', 'Avg_Hits_Last_10', 
    'Opp_GA_Avg_Last_10'
]
X = df[features]; y = df[target]
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training the XGBoost model...")
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
model.fit(X_train, y_train)
print("Model training complete!")
print("\nEvaluating model performance...")
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"  - AUC Score: {auc:.4f}")
model_filename = 'nhl_goal_predictor_model.joblib'
joblib.dump(model, model_filename)
print(f"\nTrained model saved to: {model_filename}")
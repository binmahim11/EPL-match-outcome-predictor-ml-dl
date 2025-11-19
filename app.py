
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from streamlit_lottie import st_lottie
import requests


def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# New celebration animation for correct predictions
celebration_anim = load_lottieurl(
    "https://assets10.lottiefiles.com/packages/lf20_touohxv0.json"
)

if celebration_anim is None:
    st.warning("Could not load celebration animation. Check the URL or your internet connection.")

# -------------------------------
# 1. Load and preprocess data
# -------------------------------
matches = pd.read_csv("matches.csv", index_col=0)
matches["date"] = pd.to_datetime(matches["date"])
matches["target"] = (matches["result"] == "W").astype("int")
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek

# -------------------------------
# 2. Rolling / Form Features
# -------------------------------
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group['last3_points'] = group['target'].rolling(3, closed='left').sum()
    group['last3_goals'] = group['gf'].rolling(3, closed='left').mean()
    group = group.dropna(subset=new_cols + ['last3_points','last3_goals'])
    return group

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])

# -------------------------------
# 3. Head-to-Head Win Rate
# -------------------------------
def head_to_head_win_rate(df):
    h2h = df.groupby(['team','opponent'])['target'].expanding().mean().shift(1)
    df['h2h_win_rate'] = h2h.reset_index(level=[0,1], drop=True)
    df['h2h_win_rate'] = df['h2h_win_rate'].fillna(0.5)
    return df

matches_rolling = head_to_head_win_rate(matches_rolling)

# -------------------------------
# 4. Elo Ratings
# -------------------------------
teams = matches_rolling["team"].unique()
elo = {team: 1500 for team in teams}

def update_elo(home, away, result, base_rating=1500, k=20):
    if home not in elo:
        elo[home] = base_rating
    if away not in elo:
        elo[away] = base_rating
    expected_home = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
    expected_away = 1 - expected_home
    elo[home] += k * (result - expected_home)
    elo[away] += k * ((1 - result) - expected_away)

for _, row in matches_rolling.iterrows():
    update_elo(row["team"], row["opponent"], row["target"])

matches_rolling['home_elo'] = matches_rolling['team'].map(elo)
matches_rolling['away_elo'] = matches_rolling['opponent'].map(elo)
matches_rolling['elo_diff'] = matches_rolling['home_elo'] - matches_rolling['away_elo']

# -------------------------------
# 5. Predictor Variables
# -------------------------------
predictors = ["venue_code", "opp_code", "hour", "day_code"] + new_cols + \
             ['last3_points','last3_goals','h2h_win_rate','home_elo','away_elo','elo_diff']

train = matches_rolling[matches_rolling["date"] < '2022-01-01']
test = matches_rolling[matches_rolling["date"] > '2022-01-01']

X_train = train[predictors]
y_train = train["target"]
X_test = test[predictors]
y_test = test["target"]

# -------------------------------
# 6. Define Models
# -------------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(50,25), max_iter=500, random_state=1),
}

# -------------------------------
# 7. Train Models
# -------------------------------
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    results[name] = {"predictions": preds, "accuracy": acc, "precision": prec}

# -------------------------------
# 8. Streamlit App
# -------------------------------
def main():
    st.title("⚽𝓔𝓷𝓰𝓵𝓲𝓼𝓱 𝓟𝓻𝓮𝓶𝓲𝓮𝓻 𝓛𝓮𝓪𝓰𝓾𝓮 𝓔𝓹𝓲𝓬 𝓑𝓪𝓽𝓽𝓵𝓮 🦁")
    st.write("🎯 Select a model and input two team names to see predictions:")

    model_choice = st.selectbox("Choose model", list(results.keys()))
    result = results[model_choice]

    teams_list = sorted(test['team'].unique())
    opponents_list = sorted(test['opponent'].unique())

    team_input = st.selectbox("Select Home Team", teams_list)
    opp_input = st.selectbox("Select Away Team", opponents_list)

    if team_input and opp_input:
        filtered = test[(test['team'] == team_input) & (test['opponent'] == opp_input)]
        
        if not filtered.empty:
            st.write(f"### Match: {team_input} vs {opp_input}")

            # Reset filtered index to align with predictions
            filtered = filtered.reset_index(drop=True)

            for i, row in filtered.iterrows():
                st.write(f"**Date:** {row['date'].date()}")
                st.write(f"**Actual Result (Win=1):** {row['target']}")
                st.write(f"**Predicted Result:** {result['predictions'][i]}")

                # Celebrate if actual and predicted results are the same (win)
                if result['predictions'][i] == row['target'] == 1:
                    if celebration_anim is not None:
                        st_lottie(celebration_anim, speed=1, width=200, height=200, loop=True)
                    else:
                        st.info("Prediction correctly matched a win 🎉")

                st.write("---")
        else:
            st.write("No match found for these two teams in the test dataset.")

    st.subheader(f"{model_choice} Metrics")
    st.write(f"Accuracy: {result['accuracy']:.2f}")
    st.write(f"Precision: {result['precision']:.2f}")

if __name__ == "__main__":
    main()

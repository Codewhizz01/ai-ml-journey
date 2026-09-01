import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Match Outcome Predictor", layout="centered")

# Custom styling
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
   

    .main-title {
        font-size: 62px !important;
        font-weight: 800 !important;
        color: #F1F5F9;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }
    .subtitle {
        color: #94A3B8;
        font-size: 15px;
        margin-bottom: 30px;
        line-height: 1.5;
    }
    div.stButton > button {
        background-color: #0EA5E9;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 28px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #0284C7;
        color: white;
    }
    .result-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .result-label {
        color: #94A3B8;
        font-size: 14px;
        margin-bottom: 6px;
    }
    .result-value {
        color: #F1F5F9;
        font-size: 28px;
        font-weight: 700;
    }
    .winner-tag {
        color: #22C55E;
        font-size: 12px;
        font-weight: 600;
        margin-top: 4px;
    }
    </style>
""", unsafe_allow_html=True)

model = joblib.load('../worldcup_rf_model.pkl')
feature_columns = joblib.load('../feature_columns.pkl')
team_form = pd.read_pickle('../team_form.pkl')
played_h2h = pd.read_pickle('../played_h2h.pkl')

teams = sorted(team_form['team'].unique())

st.markdown('<p class="main-title">Match Outcome Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predicts win, draw, and loss probability for international matches using rolling team form and head-to-head history, powered by a Random Forest model.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Project Notes")
    st.write("**Model:** Random Forest Classifier")
    st.write("**Features:** rolling form, form difference, head-to-head draw rate")
    st.write(f"**Teams in dataset:** {len(teams)}")
    st.divider()
    st.markdown("### Model Performance")
    st.write("**Overall accuracy:** 45%")
    st.caption("For context, random guessing across 3 outcomes (Win/Draw/Loss) would score ~33%. Draws are historically the hardest outcome to predict in football.")
    st.divider()
    st.caption("Built by Divya Bora")

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Home Team", teams, index=0)
with col2:
    team2 = st.selectbox("Away Team", teams, index=1)

if team1 == team2:
    st.warning("Please select two different teams.")
else:
    if st.button("Predict Match Outcome"):
        t1_data = team_form[team_form['team'] == team1].sort_values('date')
        t2_data = team_form[team_form['team'] == team2].sort_values('date')

        home_form = t1_data['rolling_form'].iloc[-1] if not t1_data.empty else 0
        away_form = t2_data['rolling_form'].iloc[-1] if not t2_data.empty else 0
        form_diff = home_form - away_form
        form_diff_abs = abs(form_diff)
        neutral = 0

        pair_key = '_'.join(sorted([team1, team2]))
        pair_matches = played_h2h[played_h2h['pair_key'] == pair_key]
        h2h_draw_rate = pair_matches['is_draw'].mean() if not pair_matches.empty else played_h2h['is_draw'].mean()
        num_h2h_matches = len(pair_matches)

        features = pd.DataFrame([[home_form, away_form, neutral, form_diff, form_diff_abs, h2h_draw_rate]],
                                 columns=feature_columns)
        probs = model.predict_proba(features)[0]
        result = dict(zip(model.classes_, probs))

        outcomes = {
            f"{team1} Win": result.get('H', 0),
            "Draw": result.get('D', 0),
            f"{team2} Win": result.get('A', 0)
        }
        winner = max(outcomes, key=outcomes.get)

        st.divider()
        st.markdown("#### Prediction")
        st.caption("Based on historical form and head-to-head data. Actual outcomes depend on many real-world factors not captured by this model.")

        c1, c2, c3 = st.columns(3)
        for col, (label, value) in zip([c1, c2, c3], outcomes.items()):
                    tag = '<div class="winner-tag">Most likely</div>' if label == winner else '<div class="winner-tag" style="visibility:hidden;">.</div>'
                    col.markdown(f"""
                        <div class="result-card">
                            <div class="result-label">{label}</div>
                            <div class="result-value">{value*100:.1f}%</div>
                            {tag}
                        </div>
                    """, unsafe_allow_html=True)

        st.write("")
        st.bar_chart(pd.Series(outcomes))
        st.caption(f"Based on {num_h2h_matches} past head-to-head match(es) between these teams.")

        st.divider()
        st.markdown("#### Recent Form Trend")
        form_col1, form_col2 = st.columns(2)
        with form_col1:
                    st.write(f"**{team1}**")
                    if not t1_data.empty:
                        st.line_chart(t1_data.set_index('date')['rolling_form'].tail(10))
        with form_col2:
                    st.write(f"**{team2}**")
                    if not t2_data.empty:
                        st.line_chart(t2_data.set_index('date')['rolling_form'].tail(10))
        st.divider()
        st.markdown("#### Head-to-Head History")
        if pair_matches.empty:
                st.write("No previous matches recorded between these two teams.")
        else:
                st.dataframe(
                    pair_matches[['date', 'home_team', 'away_team', 'home_score', 'away_score']]
                    .sort_values('date', ascending=False)
                    .head(10),
                    hide_index=True
                )
import pandas as pd
df=pd.read_csv('data/raw/results.csv')
 #convert data properly
df['date'] = pd.to_datetime(df['date'])

# find missing scores
print(df[df['home_score'].isnull()])
 # check duplicates
print('duplicates:',df.duplicated().sum())
#check tournament types
print(df['tournament'].value_counts().head(10))
#date range
print(df['date'].min(),df['date'].max())
#separate played matches from upcoming
played=df[df['home_score'].notnull()].copy()
upcoming=df[df['home_score'].isnull()].copy()

print("played matches:",played.shape)
print("upcoming matches:",upcoming.shape)
print(upcoming)

#1. match outcome label
played['result'] = played.apply(
    lambda row:'H' if row['home_score'] > row['away_score']
    else('A' if row['home_score'] < row['away_score'] else 'D'), axis=1)
#2 goal difference 
played['goal_diff'] = played['home_score'] - played['away_score']

#3 total goals 
played['total_goals'] = played['home_score'] + played['away_score']
print(played[['home_team' ,'away_team','home_score','away_score','result','goal_diff']].head(10))
print(played['result'].value_counts())


#sort by date first- crucial for any rolling form features
played =played.sort_values('date').reset_index(drop=True)

#quick win rate per team overall(not time-aware yet ,just a first look  )
home_wins=played[played['result']=='H']['home_team'].value_counts()
away_wins=played[played['result']=='A']['away_team'].value_counts()

print(home_wins.head(10))
print(away_wins.head(10))

home_df=played[['date','home_team','result']].rename(columns={'home_team':'team'})
home_df['points']=home_df['result'].map({'H':3 , 'D':1 ,'A':0})
away_df=played[['date','away_team','result']].rename(columns={'away_team':'team'})
away_df['points'] = away_df['result'].map({'A':3 ,'D':1,'H':0})
team_form=pd.concat([home_df,away_df]).sort_values(['team','date'])
print(team_form.head(10))


team_form['rolling_form']=(
    team_form.groupby('team')['points']
    .transform(lambda x:x.shift(1).rolling(5,min_periods=1).mean())
)
print(team_form[team_form['team']=='Abkhazia'].head(10))

#merge home team's form
played=played.merge(
    team_form[['date','team','rolling_form']],
    left_on=['date','home_team'],right_on=['date','team'],
    how='left'
).rename(columns={'rolling_form':'home_form'}).drop(columns='team')

#merge away team's form
played=played.merge(
    team_form[['date','team','rolling_form']],
    left_on=['date','away_team'],right_on=['date','team'],
    how='left'
).rename(columns={'rolling_form':'away_form'}).drop(columns='team')
print(played[['date','home_team','away_team','home_form','away_form','result']].tail(15))


# MODEL TRAINING - LOGISTIC REGRESSION AND RANDOM FOREST
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report



# form difference-captures how evenly matched two teams are
played['form_diff']=played['home_form']-played['away_form']
played['form_diff_abs'] =played['form_diff'].abs()

played['is_draw'] =(played['home_score'] == played['away_score']).astype(int)
played['pair_key'] = played.apply(
    lambda row: '_'.join(sorted([row['home_team'],row['away_team']])),
    axis=1
)
played=played.sort_values('date')
played['h2h_draw_rate']=(
    played.groupby('pair_key')['is_draw']
               .transform(lambda x: x.shift(1).expanding().mean())
)

overall_draw_rate =played['is_draw'].mean()
played['h2h_draw_rate']=played['h2h_draw_rate'].fillna(overall_draw_rate)

# Save data needed for Streamlit predictions
team_form.to_pickle('team_form.pkl')
played[['pair_key', 'is_draw', 'date', 'home_team', 'away_team', 'home_score', 'away_score']].to_pickle('played_h2h.pkl')
print("Data saved for app!")

X =played[['home_form','away_form','neutral','form_diff','form_diff_abs','h2h_draw_rate']]
Y= played['result']  # replace target with your actual result column name
 
# drop rows where feature are missing
mask=X.notna().all(axis=1)
X=X[mask]
Y=Y[mask]
 #train/test split
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,test_size=0.2,random_state=42,stratify=Y
)
#logistic regression
log_reg=LogisticRegression(max_iter=1000)
log_reg.fit(X_train,Y_train)
log_preds=log_reg.predict(X_test)
print("logistic regression accuracy:",accuracy_score(Y_test,log_preds))
print(classification_report(Y_test,log_preds))

# random forest 

rf_model=RandomForestClassifier ( n_estimators=200,
                                class_weight='balanced',
                                random_state=42)
rf_model.fit(X_train,Y_train)
rf_preds=rf_model.predict(X_test)
print("random forest accuracy:",accuracy_score(Y_test,rf_preds))
print(classification_report(Y_test,rf_preds))


# feature importance
importances=rf_model.feature_importances_
feature_name=X.columns
for name,score in zip(feature_name,importances):
    print(f"{name}:{score:.4f}")

from sklearn.model_selection import StratifiedKFold,cross_val_score

cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
cv_score = cross_val_score(rf_model,X,Y,cv=cv,scoring='accuracy')
print("CV accuracy scores:",cv_score)
print("CV mean accuracy:",cv_score.mean(),"+",cv_score.std())

match_counts = played['pair_key'].value_counts()
print(match_counts.describe())
print("pairs with only 1 match:",(match_counts==1).sum(),"out of", match_counts.shape[0])

# ---- Time-based split sanity check ----
played_sorted = played.sort_values('date').reset_index(drop=True)

X_ts = played_sorted[['home_form','away_form','neutral','form_diff','form_diff_abs','h2h_draw_rate']]
Y_ts = played_sorted['result']

mask_ts = X_ts.notna().all(axis=1)
X_ts = X_ts[mask_ts].reset_index(drop=True)
Y_ts = Y_ts[mask_ts].reset_index(drop=True)

# no shuffle - train on earlier matches, test on later ones
split_point = int(len(X_ts) * 0.8)
X_train_ts, X_test_ts = X_ts.iloc[:split_point], X_ts.iloc[split_point:]
Y_train_ts, Y_test_ts = Y_ts.iloc[:split_point], Y_ts.iloc[split_point:]

rf_ts = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf_ts.fit(X_train_ts, Y_train_ts)
rf_ts_preds = rf_ts.predict(X_test_ts)

print("Time-based split accuracy:", accuracy_score(Y_test_ts, rf_ts_preds))
print(classification_report(Y_test_ts, rf_ts_preds))

for name, score in zip(X_ts.columns, rf_ts.feature_importances_):
    print(f"{name}: {score:.4f}")

param_grid ={
    'n_estimators':[100,200,300],
    'max_depth':[5,10,15,None],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4]
}

# base model- keep class_weight='balance' like rf_ts

best_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

best_rf.fit(X_train_ts, Y_train_ts)
import joblib
joblib.dump(best_rf, 'worldcup_rf_model.pkl')
feature_columns = list(X_train_ts.columns)
joblib.dump(feature_columns, 'feature_columns.pkl')
print("Model and features saved!")
best_preds = best_rf.predict(X_test_ts)

print("model - time-based split accuracy:")
print(classification_report(Y_test_ts, best_preds))
def predict_match(team1, team2, team_form, played, feature_columns):
    # Get each team's most recent rolling form
    t1_data = team_form[team_form['team'] == team1].sort_values('date')
    t2_data = team_form[team_form['team'] == team2].sort_values('date')

    home_form = t1_data['rolling_form'].iloc[-1] if not t1_data.empty else 0
    away_form = t2_data['rolling_form'].iloc[-1] if not t2_data.empty else 0

    form_diff = home_form - away_form
    form_diff_abs = abs(form_diff)
    neutral = 0

    # Real h2h draw rate for this pair
    pair_key = '_'.join(sorted([team1, team2]))
    pair_matches = played[played['pair_key'] == pair_key]

    if not pair_matches.empty:
        h2h_draw_rate = pair_matches['is_draw'].mean()
    else:
        h2h_draw_rate = played['is_draw'].mean()  # fallback for new matchups

    features = pd.DataFrame([[home_form, away_form, neutral, form_diff, form_diff_abs, h2h_draw_rate]],
                             columns=feature_columns)

    model = joblib.load('worldcup_rf_model.pkl')
    probs = model.predict_proba(features)[0]

    return dict(zip(model.classes_, probs))
result = predict_match("Brazil", "Argentina", team_form, played, feature_columns)
print(result)











    



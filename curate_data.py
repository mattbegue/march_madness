import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm

start_date = datetime(2024,11,4)
end_date = datetime(2025,3,19)
date = start_date

base_url = "https://ncaa-api.henrygd.me"


df = pd.DataFrame()
date = start_date
for _ in tqdm(range((end_date-start_date).days + 1)):
    datestring = date.strftime('%Y/%m/%d')
    url = base_url + "/scoreboard/basketball-women/d1/" + datestring + "/all-conf"
    try:
        response = requests.request("GET", url)
        x = json.loads(response.text)

    except:
        print(f"Unable to obtain scores on date {datestring}")
        time.sleep(0.2)
        date += timedelta(days=1)
        continue

    games = x['games']
    for game in games:
        game = game['game']

        y = pd.Series()
        y['date'] = game['startDate']
        y['gameID'] = game['gameID']
        y['AwayTeam'] = game['away']['names']['short']
        y['Awayseo'] = game['away']['names']['seo']
        y['AwayScore'] = game['away']['score']
        y['HomeTeam'] = game['home']['names']['short']
        y['Homeseo'] = game['home']['names']['seo']
        y['HomeScore'] = game['home']['score']

        df = pd.concat([df, pd.DataFrame(y).T], ignore_index=True)
    df.to_pickle("/Users/mbegue/Desktop/march_madness/data/master_scores.pkl")
    time.sleep(0.2)
    date += timedelta(days=1)
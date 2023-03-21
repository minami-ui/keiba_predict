import pandas as pd
import numpy as np
import datetime
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import time
from tqdm.notebook import tqdm
import re
from urllib.request import urlopen
import pandas as pd
import time
from tqdm.notebook import tqdm

def scrape_race_results(race_id_list, pre_race_results={}):
    #race_results = pre_race_results
    race_results = pre_race_results.copy() #正しくはこちら。注意点で解説。
    for race_id in tqdm(race_id_list):
        if race_id in race_results.keys():
            continue
        try:
            time.sleep(1)
            url = "https://db.netkeiba.com/race/" + race_id
            race_results[race_id] = pd.read_html(url)[0]
        except IndexError:
            continue
#この部分は動画中に無いですが、捕捉できるエラーは拾った方が、エラーが出たときに分かりやすいです
            #except Exception as e:
            #print(e)
        #break
        #except:
        #break
    return race_results

#データ整形
def preprocessing(results):
    df = results.copy()

    # 着順に数字以外の文字列が含まれているものを取り除く
    df = df[~(df["着 順"].astype(str).str.contains("\D"))]
    df["着 順"] = df["着 順"].astype(int)

    # 性齢を性と年齢に分ける
    df["性"] = df["性齢"].map(lambda x: str(x)[0])
    df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

    # 馬体重を体重と体重変化に分ける
    df["体重"] = df["馬体重"].str.split("(", expand=True)[0].astype(int)
    df["体重変化"] = df["馬体重"].str.split("(", expand=True)[1].str[:-1].astype(int)

    # データをint, floatに変換
    df["単勝"] = df["単勝"].astype(float)

    # 不要な列を削除
    df.drop(["タイム", "着差", "調教師", "性齢", "馬体重"], axis=1, inplace=True)

    return df

if __name__ == '__main__':
    #レースIDのリストを作る
    race_id_list = []
    for year in range(2020, 2021):
        #(2008,2021)
        for place in range(6, 7):
            #(1, 11)
            for kai in range(1, 2):
                #(1,6)
                for day in range(1, 2):
                    #(1,13)
                    for r in range(1, 13):
                        #(1,13)
                        race_id = str(year).zfill(4) + str(place).zfill(2) + str(kai).zfill(2) + str(day).zfill(2) + str(r).zfill(2)
                        race_id_list.append(race_id)

    #スクレイピングしてデータを保存
    test3 = scrape_race_results(race_id_list)
    for key in test3: #.keys()は無くても大丈夫です
        test3[key].index = [key] * len(test3[key])
    results = pd.concat([test3[key] for key in test3], sort=False) 
    results.to_pickle('results.pickle')
    results = preprocessing(results)
    results.info()
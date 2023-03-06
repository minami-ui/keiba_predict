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
import tutorial1_3

''' 以下ロジスティック回帰 '''
if __name__ == '__main__':
    results = pd.read_pickle('results.pickle')
    results = tutorial1_3.preprocessing(results)
    results.info()
    
    #4着以下を全て4にする
    clip_rank = lambda x: x if x < 4 else 4
    #動画中のresultsは、preprocessing関数で前処理が行われた後のデータ
    results["rank"] = results["着 順"].map(clip_rank)
    results.drop(["着 順", "馬名"], axis=1, inplace=True)

    #カテゴリ変数をダミー変数化
    results_d = pd.get_dummies(results)

    #訓練データとテストデータに分ける
    from sklearn.model_selection import train_test_split

    X = results_d.drop(["rank"], axis=1)
    y = results_d["rank"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=0
    )

    #アンダーサンプリング
    from imblearn.under_sampling import RandomUnderSampler

    rank_1 = y_train.value_counts()[1]
    rank_2 = y_train.value_counts()[2]
    rank_3 = y_train.value_counts()[3]

    rus = RandomUnderSampler(
        #ratio={1: rank_1, 2: rank_2, 3: rank_3, 4: rank_1},
        sampling_strategy={1: rank_1, 2: rank_2, 3: rank_3, 4: rank_1},
        random_state=71
    )

    #X_train_rus, y_train_rus = rus.fit_sample(X_train.values, y_train.values)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    #訓練
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_rus, y_train_rus)

    #スコアを表示
    print(model.score(X_train, y_train), model.score(X_test, y_test))

    #予測結果を確認
    y_pred = model.predict(X_test)
    pred_df = pd.DataFrame({"pred": y_pred, "actual": y_test})
    pred_df[pred_df["pred"] == 1]["actual"].value_counts()
    
    print(pred_df)

    #回帰係数の確認
    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
    print(coefs)
    coefs[["枠 番", "馬 番", "斤量", "単勝", "人 気", "年齢", "体重", "体重変化"]]
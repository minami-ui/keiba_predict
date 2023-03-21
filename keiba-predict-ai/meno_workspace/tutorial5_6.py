#BSを使って
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

def scrape_race_info(race_id_list):
    race_infos = {}
    for race_id in tqdm(race_id_list):
        try:
            url = "https://db.netkeiba.com/race/" + race_id
            html = requests.get(url)
            html.encoding = "EUC-JP"
            soup = BeautifulSoup(html.text, "html.parser")

            texts = (
                soup.find("div", attrs={"class": "data_intro"}).find_all("p")[0].text
                + soup.find("div", attrs={"class": "data_intro"}).find_all("p")[1].text
            )
            info = re.findall(r"\w+", texts)
            info_dict = {}
            for text in info:
                if text in ["芝", "ダート"]:
                    info_dict["race_type"] = text
                if "障" in text:
                    info_dict["race_type"] = "障害"
                if "m" in text:
                    info_dict["course_len"] = int(re.findall(r"\d+", text)[0])
                if text in ["良", "稍重", "重", "不良"]:
                    info_dict["ground_state"] = text
                if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                    info_dict["weather"] = text
                if "年" in text:
                    info_dict["date"] = text
            race_infos[race_id] = info_dict
            time.sleep(1)
        except IndexError:
            continue
        except Exception as e:
            print(e)
            break
        except:
            break
    return race_infos

#date列の処理を追加
def preprocessing(results):
    df = results.copy()

    # 着 順に数字以外の文字列が含まれているものを取り除く
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
    
    df["date"] = pd.to_datetime(df["date"], format="%Y年%m月%d日")

    return df

#時系列に沿ってデータを分割
def split_data(df, test_size):
    sorted_id_list = df.sort_values("date").index.unique()
    train_id_list = sorted_id_list[: round(len(sorted_id_list) * (1 - test_size))]
    test_id_list = sorted_id_list[round(len(sorted_id_list) * (1 - test_size)) :]
    train = df.loc[train_id_list]
    test = df.loc[test_id_list]
    return train, test

def convert_binary(list, true_value):
    results = []
    for v in list:
        if v == true_value:
            results.append(1)
        else:
            results.append(0)
    return results

#前回保存したpickleファイルからデータ取得
results = pd.read_pickle('results.pickle')

# print("------------------------- result ----------------------------")
# print(results)

# #レースID一覧を取得してスクレイピング
# race_id_list = results.index.unique()
# race_infos = scrape_race_info(race_id_list)

# #DataFrame型にする
# race_infos = pd.DataFrame(race_infos).T

# race_infos.to_pickle("race_infos.pickle")

race_infos = pd.read_pickle('race_infos.pickle')

# print("------------------------- race_infos ----------------------------")
# print(race_infos)

#resultsに結合
results_addinfo = results.merge(race_infos, left_index=True, right_index=True, how="inner")

# print("------------------------- results_addinfo ----------------------------")
# print(results_addinfo)

# 前処理
results_p = preprocessing(results_addinfo)

# print("------------------------- results_p ----------------------------")
# print(results_p)

results_p.drop(["馬名"], axis=1, inplace=True)
results_d = pd.get_dummies(results_p)
results_d["rank"] = results_d["着 順"].map(lambda x: x if x < 4 else 4)

# print("------------------------- results_d ----------------------------")
# print(results_d)

train, test = split_data(results_d, test_size=0.3)
X_train = train.drop(["着 順", "date", "rank"], axis=1)
y_train = train["rank"]
X_test = test.drop(["着 順", "date", "rank"], axis=1)
y_test = test["rank"]

# 明日ここからやる

# アンダーサンプリング
# 不均衡データを均等にするための処理
from imblearn.under_sampling import RandomUnderSampler

rank_1 = train["rank"].value_counts()[1]
rank_2 = train["rank"].value_counts()[2]
rank_3 = train["rank"].value_counts()[3]

print(rank_1)

rus = RandomUnderSampler(
    sampling_strategy={1: rank_1, 2: rank_2, 3: rank_3, 4: rank_1}, random_state=71
)

X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
#fit_sample -> fit_resample

# print("------------------------- X_train_rus ----------------------------")
# print(X_train_rus)

# print("------------------------- y_train_rus ----------------------------")
# print(y_train_rus)

#ランダムフォレストによる予測モデル作成
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=100)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_test)

print("------------------------- y_pred ----------------------------")
print(y_pred)

y_pred = y_pred[:, 1] # 1位になる確率を抽出している

#ROC曲線の表示
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

#jupyterlabを使う場合、この2行はいらない
#from jupyterthemes import jtplot
#jtplot.style(theme="monokai")

# print("------------------------- X_test ----------------------------")
# print(X_test)
# print(X_test.info())
# print("------------------------- y_test ----------------------------")
# print(y_test.values.tolist())
# print(len(y_test.values.tolist()))
# print(type(y_test.values.tolist()))
# print("------------------------- y_pred ----------------------------")
# print(y_pred)
# print(type(y_pred))

roc_y_test = convert_binary(y_test.values.tolist(), 1)

fpr, tpr, thresholds = roc_curve(roc_y_test, y_pred.tolist())

plt.plot(fpr, tpr, marker="o")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.grid()
plt.show()

# AUCスコアの表示（1～0.5の間で動く、制度が高ければ1、低ければ0.5）
roc_auc_score(roc_y_test, y_pred)

y_pred_train = rf.predict_proba(X_train)[:, 1]

roc_y_train = convert_binary(y_train.tolist(), 1)

roc_auc_score(roc_y_train, y_pred_train)

# パラメータの調整
params = {
    "min_samples_split": 500,
    "max_depth": None,
    "n_estimators": 60,
    "criterion": "entropy",
    "class_weight": "balanced",
    "random_state": 100,
}

rf = RandomForestClassifier(**params)
rf.fit(X_train, y_train)
y_pred_train = rf.predict_proba(X_train)[:, 1]
y_pred = rf.predict_proba(X_test)[:, 1]

roc_y_test = convert_binary(y_test.values.tolist(), 1)
roc_y_train = convert_binary(y_train.tolist(), 1)

print("------------------------- RandomForestClassifier ----------------------------")
print("訓練データのスコア = " + str(roc_auc_score(roc_y_train, y_pred_train)))
print("テストデータのスコア = " + str(roc_auc_score(roc_y_test, y_pred)))

# 変数の重要度の表示
importances = pd.DataFrame(
    {"features": X_train.columns, "importance": rf.feature_importances_}
)
importances.sort_values("importance", ascending=False)[:20]

# LightGBMによる予測モデル作成
import lightgbm as lgb

params = {
    "num_leaves": 4,
    "n_estimators": 80,
    #'min_data_in_leaf': 15,
    "class_weight": "balanced",
    "random_state": 100,
}

lgb_clf = lgb.LGBMClassifier(**params)
lgb_clf.fit(X_train.values, y_train.values)
y_pred_train = lgb_clf.predict_proba(X_train)[:, 1]
y_pred = lgb_clf.predict_proba(X_test)[:, 1]

roc_y_train = convert_binary(y_train.tolist(), 1)
roc_y_test = convert_binary(y_test.values.tolist(), 1)

print("------------------------- LGBMClassifier ----------------------------")
print("訓練データのスコア = " + str(roc_auc_score(roc_y_train, y_pred_train)))
print("テストデータのスコア = " + str(roc_auc_score(roc_y_test, y_pred)))

# 変数の重要度の表示
importances = pd.DataFrame(
    {"features": X_train.columns, "importance": lgb_clf.feature_importances_}
)
importances.sort_values("importance", ascending=False)[:20]
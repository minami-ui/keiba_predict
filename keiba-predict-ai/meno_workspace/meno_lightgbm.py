import csv
import pandas
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import lightgbm as lgb

def read_csv(filePath):
    with open(filePath, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        return reader

def preprocessing(results):
    df = results.copy()

    # 着順に数字以外の文字列が含まれているものを取り除く
    df = df[~(df["着順"].astype(str).str.contains("\D"))]
    df["着順"] = df["着順"].astype(int)
    
    # データをint, floatに変換
    df["単勝"] = df["単勝"].astype(float)

    # 不要な列を削除
    df.drop(["タイム"], axis=1, inplace=True)
    
    df["date"] = pandas.to_datetime(df["date"], format="%Y年%m月%d日")

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

# 2020年のデータ加工
race_info = pandas.read_csv("data/race_info/2020race_info_utf8.csv")
race_result = pandas.read_csv("data/race_result/2020race_result_utf8.csv")
results_addinfo = race_result.merge(race_info, left_index=True, right_index=True, how="inner")
results_p = preprocessing(results_addinfo)
results_p.drop(["馬名"], axis=1, inplace=True)
results_d = pandas.get_dummies(results_p)
results_d["rank"] = results_d["着順"].map(lambda x: x if x < 4 else 4)

# 2021年のデータ加工
race_info_2021 = pandas.read_csv("data/race_info/2020race_info_utf8.csv")
race_result_2021 = pandas.read_csv("data/race_result/2020race_result_utf8.csv")
results_addinfo_2021 = race_result_2021.merge(race_info_2021, left_index=True, right_index=True, how="inner")
results_p_2021 = preprocessing(results_addinfo_2021)
results_p_2021.drop(["馬名"], axis=1, inplace=True)
results_d_2021 = pandas.get_dummies(results_p_2021)
results_d_2021["rank"] = results_d_2021["着順"].map(lambda x: x if x < 4 else 4)


train, test = split_data(results_d, test_size=0.3)
X_train = train.drop(["column","race_id_x", "着順", "date", "rank"], axis=1)
y_train = train["rank"]
X_test = test.drop(["column","race_id_x", "着順", "date", "rank"], axis=1)
y_test = test["rank"]
X_test_2021 = results_d_2021.drop(["column","race_id_x", "着順", "date", "rank"], axis=1)
y_test_2021 = results_d_2021["rank"]

print(X_test)

# アンダーサンプリング
# 不均衡データを均等にするための処理
from imblearn.under_sampling import RandomUnderSampler

rank_1 = train["rank"].value_counts()[1]
rank_2 = train["rank"].value_counts()[2]
rank_3 = train["rank"].value_counts()[3]

rus = RandomUnderSampler(
    sampling_strategy={1: rank_1, 2: rank_2, 3: rank_3, 4: rank_1}, random_state=0
)

X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

params = {
    "num_leaves": 5, #一つの木あたりの葉の数
    "n_estimators": 1000, #木の数
    #'min_data_in_leaf': 15,
    "class_weight": "balanced", #
    "random_state": 100,
}

lgb_clf = lgb.LGBMClassifier(**params)
lgb_clf.fit(X_train_rus.values, y_train_rus.values)
y_pred_train = lgb_clf.predict_proba(X_train_rus)[:, 1]
# 2020年のデータでテスト
y_pred = lgb_clf.predict_proba(X_test)[:, 1]

roc_y_train = convert_binary(y_train_rus.tolist(), 1)
roc_y_test = convert_binary(y_test.values.tolist(), 1)

fpr, tpr, thresholds = roc_curve(roc_y_test, y_pred.tolist())
plt.plot(fpr, tpr, marker="o")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.grid()
plt.show()

print("------------------------- LGBMClassifier ----------------------------")
print("訓練データのスコア = " + str(roc_auc_score(roc_y_train, y_pred_train)))
print("2020年テストデータのスコア = " + str(roc_auc_score(roc_y_test, y_pred)))


y_pred = lgb_clf.predict_proba(X_test_2021)[:, 1]

roc_y_train = convert_binary(y_train_rus.tolist(), 1)
roc_y_test_2021 = convert_binary(y_test_2021.values.tolist(), 1)

print("2021年テストデータのスコア = " + str(roc_auc_score(roc_y_test_2021, y_pred)))


# 変数の重要度の表示
importances = pandas.DataFrame(
    {"features": X_train_rus.columns, "importance": lgb_clf.feature_importances_}
)
importance = importances.sort_values("importance", ascending=False)

importance.to_csv("importance.csv")
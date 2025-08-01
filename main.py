"""全体の流れ
1. ビジネスの理解・課題の特定
--------------------------精度が良くなるまで繰り返す
2. データの準備・収集
3. データの理解・可視化
4. データの加工・前処理
5. モデルの作成
6. モデルの評価
--------------------------
7. レポーティング・アプリケーション化
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

"""
2. データの準備・収集
"""

# kaggleからダウンロードしたデータセットを自分のgoogleドライブに保存しておく必要がある
dir_path = "/content/drive/MyDrive/データサイエンス/titanic/"
train_df = pd.read_csv(dir_path + "train.csv")
test_df = pd.read_csv(dir_path + "test.csv")

# 今回はどの乗客が生き残ったかを予測するので、テストデータにはSurvivedカラムがない
train_df.head()
test_df.head()
print("学習データの大きさ:",train_df.shape)
print("テストデータの大きさ:",test_df.shape)
train_df.dtypes

"""
3. データの理解・可視化
"""

train_df.isnull().sum()
# train_df.info()
test_df.isnull().sum()

# 学習用データとテストデータはまとまっている方が分析しやすいみたい。業務とかでは自前でテストデータを作る？
df = pd.concat([train_df, test_df], ignore_index=True)
df.shape
df.tail()

tmp = df.groupby("Sex")["PassengerId"].count()
tmp

plt.figure(figsize=(6,4))
plt.bar(tmp.index,tmp.values)
plt.show()

tmp.plot(kind="bar", figsize=(6,4))
pass

# paletteで色付けできる。matplotlibでなくてもseabornのcountplotで集計できる
plt.figure(figsize=(6,4))
sns.countplot(data=df,x="Sex", palette="Set3")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df,x="Pclass")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=train_df,x="Survived")
plt.show()

"""
4. データの加工・前処理
"""

# 必要のないカラムを削除
df2 = df.drop(columns=["Cabin","Fare","Ticket","SibSp","Parch","Name"])

df2.head()
df2.isnull().sum()

# Embarked(港)が欠損しているレコードが2つしかないので、今回は1番数の多い港で補完する
print("欠損の数",df2["Embarked"].isnull().sum())
plt.figure(figsize=(6,4))
sns.countplot(data=df2,x="Embarked")
plt.show()
# コピーしたデータをいじった方が安全
df3 = df2.copy()
df3["Embarked"] = df3["Embarked"].fillna("S")
print(df3["Embarked"].isnull().sum())

# Age(年齢)は20代、30代が多いので、今回は中央値で保管する(実際はもっと慎重に欠損値を扱うべき)
print(df3["Age"].isnull().sum())
print(df3["Age"].min())
print(df3["Age"].max())
# 実際は20代30代が多いということは家族連れの数が多い？と推測して新しい特徴量(派生変数)を作ったりするみたい
plt.figure(figsize=(6,4))
sns.histplot(df3["Age"], kde=False, bins=8)
plt.show()
print(df3["Age"].mean())
print(df3["Age"].median())
df4 = df3.copy()
age_median = df4["Age"].median()
print(age_median)
df4["Age"] = df4["Age"].fillna(age_median)
df4["Age"].isnull().sum()

# 機械学習モデルは文字列を扱えないので、カテゴリ変数(性別、血液型など)を数値にする必要がある
# ワンホットエンコーディング(港は3つ以上種類があるので)
ohe_embarked = pd.get_dummies(df4["Embarked"],dtype=np.int8,prefix="Embarked")
ohe_embarked.head()
df5 = pd.concat([df4,ohe_embarked],axis=1)
df5.head()
# Embarked自体は必要なくなったので削除して完了
df6 = df5.drop(columns=["Embarked"])
df6.head()

# 性別は2種類しかないからラベルエンコーディング
df6["Sex"] = pd.get_dummies(df6["Sex"],dtype=np.int8,drop_first=True)
df6.head()

# 前処理の最後にモデルを作るためのデータセットを作る
# 再度学習データとテストデータに分割する(Survivedがあるかどうかで判断)
df6["Survived"].isnull()
train = df6[~df6["Survived"].isnull()]
test = df6[df6["Survived"].isnull()]
# train.head()
# test.head()
# 全nullでいらないので削除
test = test.drop(columns=["Survived"])
test.head()

# 説明変数: 目的変数を求めるために必要なカラム
# 目的変数: 予測したい値(Survived)

y_train = train["Survived"]
x_train = train.drop(columns=["Survived", "PassengerId"])

print(train.shape)
print("目的変数",y_train.shape)
print("説明変数",x_train.shape)
x_train.head()
y_train.head()

"""
5. モデルの作成
"""

# 決定木: 条件分岐を繰り返すことで予測する機械学習モデル
# 機械学習には分類と回帰(分類は生存するかどうか、回帰は株価を予測など)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)

"""
6. モデルの評価
"""


# モデルの評価手順(ホールドアウト法 -> 学習データでモデルを訓練し、テストデータで性能を評価する)
# 1 学習データを学習用と評価用に分割
# 2 分割後の学習用データだけを使ってモデルを作成
# 3 分割後の評価用データを使ってモデルを作成

from sklearn.model_selection import train_test_split

# trが学習データ、vaがテストデータ
x_tr, x_va, y_tr, y_va = train_test_split(x_train,y_train, test_size=0.2, random_state=42)
print("学習",x_tr.shape,y_tr.shape)
print("評価",x_va.shape,y_va.shape)

# 学習用と評価用のデータで目的変数(Survived)に含まれる0と1の割合が違うので、性能を上げるため割合を揃えたい
y_tr.value_counts()
y_va.value_counts()
x_tr, x_va, y_tr, y_va = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    stratify=y_train, # ここで層化することで目的変数の割合を同じにしている
    random_state=42
)
# これで大体同じ割合になっていることを確認(0の割合 / (0の割合 + 1の割合))
y_tr.value_counts()
y_va.value_counts()

# 学習データでモデルを学習
model.fit(x_tr,y_tr)

# 学習・テストデータで性能を予測
y_tr_pred = model.predict(x_tr)
y_va_pred = model.predict(x_va)
y_va_pred[:5]
y_va[:5].values

from sklearn.metrics import accuracy_score

# 基本はテストデータが改善されるようにしていく
print("テストデータの正解率:",accuracy_score(y_va,y_va_pred))  # -> 0.7932960893854749
print("学習用データの正解率:",accuracy_score(y_tr,y_tr_pred))  # -> 0.9058988764044944

# 最後にテストデータでモデルの予測をする
test.head()
x_test = test.drop(columns=["PassengerId"])
y_test_pred = model.predict(x_test)
y_test_pred[:5]

# テストデータにモデルの予測値を反映
test["Survived"] = y_test_pred
test.head()

submit_df = test[["PassengerId", "Survived"]].set_index("PassengerId")
submit_df = submit_df["Survived"].astype(int)
submit_df.head()

# kaggleにアップロードして最終的なモデル評価するためにcsvにする
submit_df.to_csv("submit.csv")

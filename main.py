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

"""
4. データの加工・前処理
"""

"""
5. モデルの作成
"""

"""
6. モデルの評価
"""

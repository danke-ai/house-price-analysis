# -*- coding: utf-8 -*-
# 房价影响因素分析与预测 - 完整代码

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ========== 1. 读数据 ==========
df = pd.read_csv("kc_house_data.csv")
print("数据规模:", df.shape)

# ========== 2. 探索 ==========
print("\n价格统计：")
print(df["price"].describe())

# ========== 3. 相关性分析 ==========
numeric = df.select_dtypes(include=["number"])
corr = numeric.corr()
print("\n与价格最相关的特征：")
print(corr["price"].sort_values(ascending=False).head(8))

# ========== 4. 特征工程 ==========
df["age"] = df["yr_built"].apply(lambda x: 2026 - x)   # 房龄
df["price_wan"] = df["price"] / 10000                   # 万元

# ========== 5. 线性回归建模 ==========
features = ["sqft_living", "bedrooms", "bathrooms", "grade", "waterfront", "view"]
X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n模型评估：")
print("R² =", round(r2_score(y_test, y_pred), 4))

# ========== 6. 特征重要性 ==========
print("\n各特征对价格的影响（系数）：")
for name, coef in zip(features, model.coef_):
    print("  %-13s %+.1f" % (name, coef))

# 房价影响因素分析与预测

## 项目简介
分析"房价由什么决定"，建立线性回归模型预测房价。

## 数据
- `kc_house_data.csv`：21613 条房产交易记录 × 21 个特征

## 方法
- 数据探索（pandas）
- 相关性分析（seaborn 热力图）
- 特征工程（构造房龄、房龄分级等新特征）
- 线性回归建模（scikit-learn，80% 训练 / 20% 测试）

## 结果
- **R² = 0.60**，能解释房价 60% 的波动
- 识别出 **3 大价格影响因素**：
  - 海景（+55.7 万）
  - 房屋评级（+9.1 万/级）
  - 面积（+197 /平方英尺）

## 技术栈
Python · pandas · seaborn · matplotlib · scikit-learn

## 可视化仪表板
<img width="536" height="298" alt="dashboard" src="https://github.com/user-attachments/assets/3d85982c-6985-402b-9f5c-64d943d2261d" />

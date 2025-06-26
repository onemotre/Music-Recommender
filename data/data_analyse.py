import pandas as pd
import numpy as np

df = pd.read_csv(r'./Spotify_Youtube.csv')
# 查看前几行数据
print("数据前5行:")
print(df.head())

# 查看后几行数据
print("\n数据后5行:")
print(df.tail())

# 查看数据形状(行数和列数)
print("\n数据形状(行数, 列数):")
print(df.shape)

# 查看列名
print("\n列名:")
print(df.columns.tolist())

# 查看数据类型
print("\n数据类型:")
print(df.dtypes)

# 查看基本信息摘要
print("\n数据基本信息:")
print(df.info())

# 数值型数据的统计描述
print("\n数值型数据统计描述:")
print(df.describe())

# 包括非数值型数据的统计描述
print("\n所有数据统计描述:")
print(df.describe(include='all'))

# 查看每列的唯一值数量
print("\n每列唯一值数量:")
print(df.nunique())

# 检查每列的缺失值数量
print("\n缺失值数量:")
print(df.isnull().sum())

# 缺失值占比
print("\n缺失值占比:")
print(df.isnull().mean().round(4) * 100, '%')

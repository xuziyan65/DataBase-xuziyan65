import pandas as pd
import mysql.connector

# 1. 读 Excel（中文列名会被原样读入 DataFrame）
df = pd.read_csv(r"C:\Users\m1774\Downloads\Copy of 碧桂园产品价格清单2023.05.28(1)(2).csv")

# 清洗列名
df.columns = df.columns.str.replace(r'\s+', '', regex=True)

# 假设这些是金额相关的列名
money_cols = [
    "销售单价（不含税）", "采购价（不含税）", "项目价（不含税）", "一个月账期价格"
]

for col in money_cols:
    if col in df.columns:
        # 去掉非数字和小数点的字符，并转为float
        df[col] = df[col].astype(str).str.replace(r'[^\d\.]', '', regex=True).replace('', '0').astype(float)

# 假设这些是利润率相关的列名
percent_cols = ["采购价利润率", "项目价利润率", "一个月账期价格利润率"]  # 根据你的表头补全

for col in percent_cols:
    if col in df.columns:
        # 去掉百分号，转为float
        df[col] = df[col].astype(str).str.replace('%', '').replace('', '0').astype(float)

# 替换所有 NaN 为 0
df = df.fillna({col: 0 for col in df.select_dtypes(include=['float', 'int']).columns})

# 2. 取出所有列名列表（原样保留中文）
cols = list(df.columns)

print(cols)

# 3. 拼 SQL 语句
# D   - 列名用反引号包裹，防止特殊字符
#    - 占位符数量和列数一致
columns_str   = ", ".join(f"`{c}`" for c in cols)
placeholders  = ", ".join(["%s"] * len(cols))
sql = f"INSERT INTO product ({columns_str}) VALUES ({placeholders})"

# 4. 连接 MySQL（确保 product 表已经用相同中文列名建好，且表字符集为 utf8mb4）
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="yjc1234560",
    database="my_database",
    charset="utf8mb4"
)
cursor = conn.cursor()

# 5. 批量插入
for _, row in df.iterrows():
    values = [ row[col] for col in cols ]
    cursor.execute(sql, values)

conn.commit()
cursor.close()
conn.close()
print("✅ 所有中文列名数据已插入完成")

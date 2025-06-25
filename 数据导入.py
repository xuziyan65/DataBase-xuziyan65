import pandas as pd
from sqlalchemy import create_engine

# 数据库连接
db_user = 'root'
db_password = 'xzy20010506'
db_host = 'localhost'
db_port = '3306'

db_name = 'my_database'
engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# 读取Excel
df = pd.read_excel('Copy of new 印尼25年最新价格库(不含PE).XLSX', sheet_name='正常销售价格（面价下浮）')

# 重命名列名，确保和数据库字段一致
df = df.rename(columns={
    'Material': 'Material',
    'Describrition': 'Description',
    '面价': '面价',
    '折扣': '折扣',
    '币种': '币种',
    '出厂价_含税)': '出厂价_含税',
    '出厂价_不含税': '出厂价_不含税',
    '仓位': '仓位',
    '仓位描述': '仓位描述',
    '产品分类': '产品分类',
    '维护单位': '维护单位',
    '货币': '货币',
    '维护给的客户': '维护给的客户',
    '是否参与返点': '是否参与返点',
    '流程编码': '流程编码',
    '有效期到': '有效期止'
})

# 去除所有列名首尾空格
df.columns = df.columns.str.strip()

# 先转换为 datetime 类型，自动识别数字和字符串
df['有效期止'] = pd.to_datetime(df['有效期止'], errors='coerce')

# 再格式化为字符串（MySQL DATE 格式）
df['有效期止'] = df['有效期止'].dt.strftime('%Y-%m-%d')

# 导入到MySQL（如果表已存在，append；如果要覆盖用replace）
df.to_sql('product', con=engine, if_exists='append', index=False)


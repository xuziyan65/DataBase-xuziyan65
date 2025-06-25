import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import re

# 数据库连接配置
db_user = 'root'
db_password = 'yjc1234560'
db_host = 'localhost'
db_port = '3306'
db_name = 'my_database'
engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Streamlit 页面标题
st.title("批量报价查询")

mm_to_inch = {
    "20": '1/2"', "25": '3/4"', "32": '1"', "50": '1-1/2"', "63": '2"', "75": '2-1/2"', "90": '3"', "110": '4"'
}
# 反向字典
inch_to_mm = {v: k for k, v in mm_to_inch.items()}

def normalize_text(s):
    # 全部转小写，去掉非字母数字的符号
    return re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', '', str(s).lower())

def all_keywords_in_text(keywords, text):
    # 检查所有关键字是否都在目标文本中
    return all(k in text for k in keywords)

def expand_spec_numbers(numbers):
    expanded = set(numbers)
    for n in numbers:
        n = n.replace('"', '')  # 去掉引号
        # mm to inch
        if n in mm_to_inch:
            expanded.add(mm_to_inch[n].replace('"', ''))
        # inch to mm
        if n + '"' in inch_to_mm:
            expanded.add(inch_to_mm[n + '"'])
        if n in inch_to_mm:
            expanded.add(inch_to_mm[n])
    return expanded

uploaded_file = st.file_uploader("上传Excel或CSV文件", type=["xlsx", "csv"])
if uploaded_file:
    # 读取文件
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.write("文件预览：", df.head())

    # 让用户选择"名称"和"规格型号"列
    columns = df.columns.tolist()
    name_col = st.selectbox("请选择'名称'列", columns)
    spec_col = st.selectbox("请选择'规格型号'列", columns)
    # 数量列可选
    qty_col = st.selectbox("请选择'数量'列（可选）", ["无"] + columns)
    # 价格种类选择
    price_type = st.selectbox("请选择价格种类", ["出厂价_含税", "出厂价_不含税"])

    if st.button("查询报价"):
        results = []
        for idx, row in df.iterrows():
            name = row[name_col]
            spec = row[spec_col]
            qty = row[qty_col] if qty_col != "无" else 1
            try:
                qty = float(qty)
            except:
                qty = 1
            # 先查出所有可能的行（不加条件，后续用Python筛选）
            query = "SELECT * FROM product"
            db_results = pd.read_sql(text(query), engine)

            # 预处理用户输入名称关键字
            user_keywords = list(str(name).lower().replace('-', '').replace(' ', ''))
            # 预处理用户输入规格型号数字片段
            def extract_numbers(s):
                return re.findall(r'\d+\/\d+|\d+\.\d+|\d+', str(s))
            spec_numbers = extract_numbers(spec)
            spec_numbers_expanded = expand_spec_numbers(spec_numbers)

            match_count = 0
            for i, db_row in db_results.iterrows():
                db_spec = db_row["Description"] if "Description" in db_row else ""
                db_spec_norm = normalize_text(db_spec)
                db_spec_numbers = extract_numbers(db_spec)
                db_spec_numbers_expanded = expand_spec_numbers(db_spec_numbers)
                # 名称关键字和规格型号数字片段都要匹配
                if all_keywords_in_text(user_keywords, db_spec_norm) and (spec_numbers_expanded & db_spec_numbers_expanded):
                    price = db_row[price_type] if price_type in db_row else 0
                    total = price * qty
                    results.append({
                        "名称": name,
                        "规格型号": spec,
                        "Description": db_row.get("Description", ""),
                        "单价": price,
                        "数量": qty,
                        "总价": total,
                        "匹配标识": "名称+数字片段智能匹配"
                    })
                    match_count += 1
        if results:
            final_df = pd.DataFrame(results)
            # 统计每个名称出现的次数
            name_counts = final_df['名称'].value_counts()
            # 标红有多个匹配的名称
            def highlight_name(val):
                if name_counts.get(val, 0) > 1:
                    return 'color: red; font-weight: bold;'
                return ''
            st.write("查询结果：")
            st.dataframe(final_df.style.applymap(highlight_name, subset=['名称']))
            st.success(f"总价合计：{final_df['总价'].sum():,.2f}")
        else:
            st.info("没有查到相关报价。")

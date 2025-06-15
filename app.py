import streamlit as st
import mysql.connector
import pandas as pd

# Streamlit 页面标题
st.title("产品自动报价系统")

# 1. 输入SAP编号（支持多个）
sap_ids_input = st.text_input(
    "请输入一个或多个 SAP 编号（用逗号或空格分隔）：",
    key="sap_ids_input"
)

# 2. 选择价格类型
price_type = st.radio(
    "请选择价格类型",
    ("采购价（不含税）", "项目价（不含税）", "一个月账期价格"),
    key="price_type"
)

# 3. 查询按钮
if st.button("查询"):
    sap_ids = [x.strip() for x in st.session_state["sap_ids_input"].replace('，', ',').replace(' ', ',').split(',') if x.strip()]
    if not sap_ids:
        st.warning("请输入至少一个 SAP 编号")
    else:
        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="yjc1234560",
                database="my_database",
                port=3306,
                charset="utf8mb4"
            )
            cursor = conn.cursor(dictionary=True)
            format_strings = ','.join(['%s'] * len(sap_ids))
            query = f"SELECT * FROM product WHERE `SAP编号` IN ({format_strings})"
            cursor.execute(query, tuple(sap_ids))
            result = cursor.fetchall()
            cursor.close()
            conn.close()

            if result:
                df = pd.DataFrame(result)
                st.session_state["query_result"] = df
                st.session_state["queried_price_type"] = st.session_state["price_type"]
            else:
                st.session_state["query_result"] = None
                st.session_state["queried_price_type"] = st.session_state["price_type"]
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state["query_result"] = None
            st.session_state["queried_price_type"] = st.session_state["price_type"]

# 4. 显示查询结果和输入控件（只要 session_state 里有结果就显示）
if "query_result" in st.session_state and st.session_state["query_result"] is not None:
    df = st.session_state["query_result"]
    queried_price_type = st.session_state.get("queried_price_type", st.session_state["price_type"])
    st.write("请为每个产品输入数量：")
    quantities = []
    prices = []
    for i, row in df.iterrows():
        sap = row['SAP编号']
        qty_key = f"qty_{sap}"
        price_key = f"price_{sap}"
        # 只在 session_state 没有时初始化
        if qty_key not in st.session_state:
            st.session_state[qty_key] = 1
        if price_key not in st.session_state:
            default_price = row[queried_price_type] if pd.notnull(row[queried_price_type]) else 0
            st.session_state[price_key] = float(default_price)
        col1, col2 = st.columns(2)
        with col1:
            qty = st.number_input(
                f"{row['名称']}（SAP编号: {sap}）数量",
                min_value=0,
                key=qty_key
            )
        with col2:
            price = st.number_input(
                f"{row['名称']}（SAP编号: {sap}）单价",
                min_value=0.0,
                key=price_key
            )
        quantities.append(qty)
        prices.append(price)
    df["输入数量"] = quantities
    df["输入单价"] = prices
    df["小计"] = df["输入数量"] * df["输入单价"]
    show_cols = ["序号", "系列", "SAP编号", "名称", "规格型号", "单位", "输入数量", "输入单价", "小计"]
    st.dataframe(df[show_cols])
    total = df["小计"].sum()
    st.success(f"所选产品的总价为：{total:.2f}")
import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from pathlib import Path
from collections import Counter
from openai import OpenAI
from search_utils import (format_row, search_with_keywords, expand_keyword_with_synonyms, classify_tokens, expand_token_with_synonyms_and_units,
    normalize_material, split_with_synonyms, get_synonym_words, expand_unit_tokens, get_db_engine, load_data, insert_product, delete_products,
    ai_select_best_with_gpt)
import hashlib

# — 页面配置：宽屏布局、标题 —
st.set_page_config(
    page_title="产品报价系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# — 自定义 CSS —
st.markdown("""
<style>
/* 主容器卡片，最大宽度更大 */
.block-container {
    max-width: 1600px !important;
    margin: 2rem auto !important;
    background: #fff !important;
    padding: 2rem !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
}
/* 标题颜色 */
h1, h2 {
    color: #333 !important;
}
/* 按钮美化 */
.stButton>button {
    background-color: #005B96 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.5em 1.5em !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
}
.stButton>button:hover {
    background-color: #004173 !important;
}
</style>
""", unsafe_allow_html=True)

# — 通用设置 & 数据库连接 —

engine = get_db_engine()

# — Session State 初始化 —
for k, default in [("cart",[]),("last_out",pd.DataFrame()),("to_cart",[]),("to_remove",[])]:
    if k not in st.session_state:
        st.session_state[k] = default
#把用户选中的查询结果条目加入购物车，并清空本次选择，支持多选批量添加
def add_to_cart():
    for i in st.session_state.to_cart:
        st.session_state.cart.append(st.session_state.last_out.loc[i].to_dict())
    # 清空选择（推荐用 pop 或 del）
    if "to_cart" in st.session_state:
        del st.session_state["to_cart"]

#删除购物车中的条目，支持多选批量删除
def remove_from_cart():
    idxs = set(st.session_state.to_remove)
    st.session_state.cart = [it for j,it in enumerate(st.session_state.cart) if j not in idxs]
    if "to_remove" in st.session_state:
        del st.session_state["to_remove"]

# — 侧边栏导航 —
st.sidebar.header("导  航")
page = st.sidebar.radio("操作", ["查询产品", "批量查询", "添加产品", "删除产品"])
st.sidebar.markdown("---")
st.sidebar.caption("Powered by Streamlit")


# 页面切换和主逻辑
if page == "查询产品":
    st.header("产品查询系统")
    df = load_data()
#输入框布局
    c1, c3 = st.columns([6,1])
    with c1:
        keyword = st.text_input(
            "关键词（名称、规格、材质可一起输入）",
            key="keyword"
        )
    with c3:
        qty = st.number_input(
            "数量", min_value=1, key="qty"
        )
    mat_kw = st.text_input(
        "物料号搜索", key="mat_kw"
    )
    price_type = st.selectbox(
        "价格字段", ["出厂价_含税","出厂价_不含税"],
        key="price_type"
    )
    fuzzy_mode = st.checkbox(
        "未查到结果时启用模糊查找（并显示匹配度）",
        key="fuzzy_mode"
    )
    debug_mode = st.checkbox("开启调试模式 (显示关键词解析结果)", key="debug_mode")

    #查询按钮
    query_c1, query_c2, _ = st.columns([2, 2, 8])

    with query_c1:
        if st.button("查询", use_container_width=True):
            keyword = st.session_state.get("keyword", "").strip()

            # 如果开启调试模式，则显示解析结果
            if st.session_state.get("debug_mode", False) and keyword:
                with st.expander("🔍 调试信息：关键词解析结果", expanded=True):
                    _, _, chinese_tokens = classify_tokens(keyword)
                    st.write("**原始输入:**")
                    st.code(keyword, language=None)
                    st.write("**归一化后 (用于部分匹配):**")
                    st.code(normalize_material(keyword), language=None)
                    st.write("**最终解析出的 Tokens (用于搜索):**")
                    st.write(chinese_tokens)
                    st.info("提示：搜索时会用上面的 Tokens 去匹配数据库中的产品描述。请检查 Tokens 是否符合您的预期。")
                st.markdown("---")

            out_df = pd.DataFrame()
            qty = st.session_state.qty if "qty" in st.session_state else 1
            
            # 根据价格字段选择，动态决定要显示的列
            base_cols = ["Material", "Describrition", "Describrition_English", "数量"]
            price_col = st.session_state.price_type
            show_cols = base_cols + [price_col]

            # 优先物料号精确查找
            mat_kw = st.session_state.get("mat_kw", "").strip()
            if mat_kw:
                filtered = df[df["Material"].astype(str).str.contains(mat_kw)]
                if not filtered.empty:
                    out_df = pd.DataFrame(filtered.copy())  # 强制DataFrame
                    out_df["数量"] = qty
                    out_df = out_df[[col for col in show_cols if col in out_df.columns]]
                    st.session_state.last_out = out_df
                else:
                    st.session_state.last_out = pd.DataFrame()
                    st.warning("⚠️ 未查询到符合条件的产品")
            else:
                # 原有关键词查找逻辑
                results = search_with_keywords(df, st.session_state.keyword, "Describrition", strict=True)
                #模糊查询
                if not results and st.session_state.fuzzy_mode:
                    fuzzy_results = search_with_keywords(df, st.session_state.keyword, "Describrition", strict=False, return_score=True)
                    if fuzzy_results:
                        out_df = pd.DataFrame([r[0] for r in fuzzy_results])
                        out_df["匹配度"] = [round(r[1], 2) for r in fuzzy_results]
                        out_df = out_df.sort_values("匹配度", ascending=False)
                        out_df["数量"] = qty
                        show_cols_fuzzy = show_cols + ["匹配度"]
                        out_df = out_df[[col for col in show_cols_fuzzy if col in out_df.columns]]

                        # -- 修改：直接返回所有模糊查询结果，而不是只显示前三名匹配度的结果 --
                        st.session_state.last_out = out_df
                    else:
                        st.session_state.last_out = pd.DataFrame()
                        st.warning("⚠️ 未查询到符合条件的产品")
                #精准查询
                elif results:
                    out_df = pd.DataFrame(results)
                    out_df["数量"] = qty
                    out_df = out_df[[col for col in show_cols if col in out_df.columns]]
                    st.session_state.last_out = out_df
                else:
                    st.session_state.last_out = pd.DataFrame()
                    st.warning("⚠️ 未查询到符合条件的产品")

    with query_c2:
        # The AI button is only active if there are fuzzy results to choose from
        can_ai_select = (
            "last_out" in st.session_state and
            not st.session_state.last_out.empty and
            "匹配度" in st.session_state.last_out.columns
        )
        if st.button("🤖 AI 优选", use_container_width=True, disabled=not can_ai_select):
            with st.spinner("🤖 AI 正在分析最佳匹配..."):
                top_3_df = st.session_state.last_out.head(3)
                if isinstance(top_3_df, pd.DataFrame):
                    best_choice_df, message = ai_select_best_with_gpt(
                        st.session_state.keyword, top_3_df
                    )
                else:
                    best_choice_df, message = None, "数据类型错误"
            
            if best_choice_df is not None:
                # Add to cart
                item_to_add = best_choice_df.iloc[0].to_dict()
                st.session_state.cart.append(item_to_add)
                st.success("✅ AI已为您选择产品并加入购物车！")
                st.rerun() # To refresh cart view
            else:
                st.error(message)


    # 查询结果展示和购物车操作（无论是否刚点了查询按钮，只要有结果都显示）
    out_df = st.session_state.get("last_out", pd.DataFrame())
    if not out_df.empty and isinstance(out_df, pd.DataFrame):
        # 联塑优先排序
        from search_utils import prioritize_liansu
        out_df = prioritize_liansu(out_df)
        st.dataframe(out_df, use_container_width=True)
        to_cart = st.multiselect(
            "选择要加入购物车的行",
            options=list(out_df.index),
            format_func=lambda i: format_row(i, out_df),
            key="to_cart"
        )
        if st.button("添加到购物车", key="add_cart"):
            for i in to_cart:
                st.session_state.cart.append(out_df.loc[i].to_dict())
            if "to_cart" in st.session_state:
                del st.session_state["to_cart"]
            st.success("✅ 已加入购物车")

    # 购物车只在有内容时显示
    if st.session_state.cart:
        cart_df = pd.DataFrame(st.session_state.cart)
        st.dataframe(cart_df, use_container_width=True)
        to_remove = st.multiselect(
            "选择要删除的购物车条目",
            options=list(cart_df.index),
            format_func=lambda i: cart_df.loc[i, "产品描述"] if "产品描述" in cart_df.columns else cart_df.loc[i, "Describrition"],
            key="to_remove"
        )
        if st.button("删除所选", key="del_cart_bottom"):
            idxs = set(to_remove)
            st.session_state.cart = [it for j, it in enumerate(st.session_state.cart) if j not in idxs]
            if "to_remove" in st.session_state:
                del st.session_state["to_remove"]
            st.rerun()

elif page == "批量查询":
    st.header("📦 批量导入查询")
    st.info("请上传一个 Excel (.xlsx) 或 CSV (.csv) 文件。文件中需要包含 **名称**、**规格** 和 **数量** 列。")

    uploaded_file = st.file_uploader(
        "上传查询文件",
        type=["xlsx", "csv"],
        key="batch_file_uploader"
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        # 只要内容变了就重新读取
        if (
            'query_df' not in st.session_state
            or st.session_state.get('uploaded_file_hash') != file_hash
        ):
            try:
                if uploaded_file.name.endswith('.csv'):
                    from io import StringIO
                    st.session_state.query_df = pd.read_csv(StringIO(file_bytes.decode('utf-8')))
                else:
                    from io import BytesIO
                    st.session_state.query_df = pd.read_excel(BytesIO(file_bytes))
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.uploaded_file_hash = file_hash
            except Exception as e:
                st.error(f"读取文件时出错: {e}")
                st.stop()
        query_df = st.session_state.query_df
        file_columns = query_df.columns.tolist()

        st.markdown("---")
        st.subheader("请为查询指定列")

        c1, c2, c3 = st.columns(3)
        with c1:
            name_col = st.selectbox("名称所在列", options=file_columns, key="batch_name_col")
        with c2:
            spec_col = st.selectbox("规格所在列", options=file_columns, key="batch_spec_col")
        with c3:
            quantity_col = st.selectbox("数量所在列", options=file_columns, key="batch_quantity_col")


        if st.button("🚀 开始批量查询", use_container_width=True):
            st.session_state.cart = []  # 这行是关键
            
            # --- START: 诊断代码 ---
            st.info("--- 诊断信息 ---")
            st.write(f"👉 您选择的物资名称列: **{name_col}**")
            st.write(f"👉 您选择的规格列: **{spec_col}**")
            
            if not query_df.empty:
                first_row = query_df.iloc[0]
                first_row_name = str(first_row.get(name_col, "【读取失败】"))
                first_row_spec = str(first_row.get(spec_col, "【读取失败】"))
                
                st.write(f"👉 读取到表格第一行的名称: **{first_row_name}**")
                st.write(f"👉 读取到表格第一行的规格: **{first_row_spec}**")
                st.write(f"👉 根据第一行生成的关键词: **{first_row_name.strip()} {first_row_spec.strip()}**")

            st.info("--- 诊断结束 ---")
            # --- END: 诊断代码 ---

            # 数据帧已加载，列名已选择。我们可以直接开始处理。
            products_df = load_data()
            results_log = []
            
            progress_bar = st.progress(0, text="正在准备批量查询...")
            total_rows = len(query_df)
            
            with st.spinner("正在逐条查询并使用 AI 优选，请稍候..."):
                for index, row in query_df.iterrows():
                    progress_text = f"正在处理: {int(str(index)) + 1}/{total_rows}"
                    progress_bar.progress((int(str(index)) + 1) / total_rows, text=progress_text)
                    
                    # Combine name and spec, then clean it
                    try:
                        name_val = str(row[name_col]) if pd.notna(row[name_col]) else ""
                    except Exception:
                        name_val = ""
                    try:
                        spec_val = str(row[spec_col]) if pd.notna(row[spec_col]) else ""
                    except Exception:
                        spec_val = ""
                    
                    # 关键修正：直接合并，不再进行独立的标点清理。
                    # 所有的清理和解析都统一由 search_with_keywords 函数处理，以保证逻辑一致。
                    keyword = f"{name_val} {spec_val}".strip()
                    
                    # 检查是否需要人工核查
                    check_msg = ""

                    # Ensure quantity is a valid number, default to 1 if not
                    val = row.get(quantity_col, 1)
                    try:
                        quantity = int(val) if val is not None and val != "" else 1
                    except Exception:
                        quantity = 1


                    best_choice_df = None
                    status = "未找到"

                    # Step 1: Strict search
                    strict_results = search_with_keywords(products_df, keyword, "Describrition", strict=True)
                    
                    if strict_results:
                        candidates_df = pd.DataFrame(strict_results)
                        # Use AI to select from strict results (take top 5 to be safe with token limits)
                        top_3_df = candidates_df.head(5)
                        if isinstance(top_3_df, pd.DataFrame):
                            best_choice_df, message = ai_select_best_with_gpt(keyword, top_3_df)
                        else:
                            best_choice_df, message = None, "数据类型错误"
                        if message == "Success" and best_choice_df is not None and not best_choice_df.empty:
                            status = "✅ AI从严格匹配结果中选择"
                    
                    # Step 2: Fuzzy search if strict search gave no result for AI
                    if best_choice_df is None or best_choice_df.empty:
                        fuzzy_results = search_with_keywords(products_df, keyword, "Describrition", strict=False, return_score=True)
                        if fuzzy_results:
                            fuzzy_df = pd.DataFrame([r[0] for r in fuzzy_results])
                            fuzzy_df["匹配度"] = [r[1] for r in fuzzy_results]
                            fuzzy_df = fuzzy_df.sort_values("匹配度", ascending=False)
                            
                            # Use AI to select from top 3 fuzzy results
                            top_3_df = fuzzy_df.head(3)
                            if isinstance(top_3_df, pd.DataFrame):
                                best_choice_df, message = ai_select_best_with_gpt(keyword, top_3_df)
                            else:
                                best_choice_df, message = None, "数据类型错误"
                            if message == "Success" and best_choice_df is not None and not best_choice_df.empty:
                                status = "🟡 AI从模糊匹配结果中选择"

                    # Step 3: Add to cart if AI made a selection
                    if best_choice_df is not None and not best_choice_df.empty:
                        selected_item = best_choice_df.iloc[0].to_dict()
                        selected_item['数量'] = quantity
                        # 新增：AW给水或D排水人工核查提示
                        descr = selected_item.get('Describrition', '')
                        if ("AW给水" in descr) or ("D排水" in descr):
                            selected_item['人工核查提示'] = "该产品为AW给水或D排水，需要二次人工核查"
                        st.session_state.cart.append(selected_item)
                        results_log.append({
                            "查询关键词": keyword,
                            "匹配结果": selected_item.get("Describrition", "N/A"),
                            "状态": status
                        })
                    else:
                        # 构造一个“未找到”占位字典，字段与购物车其它条目一致
                        not_found_item = {
                            "Material": "无",
                            "Describrition": f"未找到：{keyword}",
                            "Describrition_English": "",
                            "数量": quantity
                            # 你可以根据实际表结构补充其它字段
                        }
                        if check_msg:
                            not_found_item['人工核查提示'] = check_msg
                        st.session_state.cart.append(not_found_item)
                        results_log.append({
                            "查询关键词": keyword,
                            "匹配结果": "---",
                            "状态": "❌ 未找到或AI无法选择"
                        })

            progress_bar.empty()
            st.success(f"🎉 批量查询完成！")
            
            # Display results log
            st.subheader("批量查询结果日志")
            if results_log:
                st.dataframe(pd.DataFrame(results_log), use_container_width=True)
            
            # Rerun to update the cart display on the main page if needed,
            # but showing it here might be better ux
            if st.session_state.cart:
                st.subheader("🛒 当前购物车")
                cart_df = pd.DataFrame(st.session_state.cart)
                show_cols = [
                    "序号","Material","Describrition","Describrition_English", "出厂价_含税","出厂价_不含税","匹配度","人工核查提示"
                ]
                # 只保留存在于cart_df中的列
                show_cols = [col for col in show_cols if col in cart_df.columns]
                st.dataframe(cart_df[show_cols], use_container_width=True)

elif page == "添加产品":
    st.header(" 添加新产品到数据库")

    # 1. 登录状态标志
    if "add_product_logged_in" not in st.session_state:
        st.session_state.add_product_logged_in = False

    # 2. 如果未登录，显示登录表单
    if not st.session_state.add_product_logged_in:
        with st.form("add_product_login_form"):
            username = st.text_input("账户", key="add_product_username")
            password = st.text_input("密码", type="password", key="add_product_password")
            login_submitted = st.form_submit_button("登录")
        if login_submitted:
            # 这里用你自己的用户名和密码校验逻辑
            if username == "vantsing" and password == "vantsing2020":  # 替换为你的账号密码
                st.session_state.add_product_logged_in = True
                st.success("登录成功！")
                st.rerun()
            else:
                st.error("账户或密码错误，请重试。")
        st.stop()  # 阻止后续内容渲染
    # 3. 已登录，显示原有添加产品表单
    df0 = load_data()
    cols = df0.columns.tolist()

    with st.form("add_form"):
        new_vals = {}
        for col in cols:
            if col == "序号":
                continue
            label = col + ("（必填）" if col in ["Describrition","出厂价_含税","出厂价_不含税"] else "")
            dtype = df0[col].dtype
            if col in ["出厂价_含税","出厂价_不含税"]:
                new_vals[col] = st.text_input(label, key=f"add_{col}")
            elif pd.api.types.is_integer_dtype(dtype):
                new_vals[col] = st.number_input(label, step=1, format="%d", key=f"add_{col}")
            elif pd.api.types.is_float_dtype(dtype):
                new_vals[col] = st.number_input(label, format="%.2f", key=f"add_{col}")
            else:
                new_vals[col] = st.text_input(label, key=f"add_{col}")

        submitted = st.form_submit_button("提交新增")

    if submitted:
        missing = [
            f for f in ["Describrition","出厂价_含税","出厂价_不含税"]
            if not new_vals.get(f) or (isinstance(new_vals[f], str) and not new_vals[f].strip())
        ]
        if missing:
            st.error(f"⚠️ 以下字段为必填：{', '.join(missing)}")
        else:
            insert_product(new_vals)
            load_data.clear()
            st.success("✅ 产品已添加到数据库！")

else:
    st.header("🗑️ 删除产品")

    # 1. 登录状态标志
    if "add_product_logged_in" not in st.session_state:
        st.session_state.add_product_logged_in = False

    # 2. 如果未登录，显示登录表单
    if not st.session_state.add_product_logged_in:
        with st.form("delete_product_login_form"):
            username = st.text_input("账户", key="delete_product_username")
            password = st.text_input("密码", type="password", key="delete_product_password")
            login_submitted = st.form_submit_button("登录")
        if login_submitted:
            if username == "vantsing" and password == "vantsing2020":  # 替换为你的账号密码
                st.session_state.add_product_logged_in = True
                st.success("登录成功！")
                st.rerun()
            else:
                st.error("账户或密码错误，请重试。")
        st.stop()  # 阻止后续内容渲染
    # 3. 已登录，显示原有删除产品界面
    df = load_data()
    if df.empty:
        st.info("当前无产品可删除。")
    else:
        materials = st.multiselect(
            "请选择要删除的产品 (Material)",
            options=df["Material"].tolist(),
            format_func=lambda m: str(m)
        )
        if st.button("删除选中产品"):
            delete_products(materials)
            load_data.clear()
            st.success("✅ 删除成功！")

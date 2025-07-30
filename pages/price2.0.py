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
    ai_select_best_with_gpt,prioritize_liansu)
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

# 替换功能的session state
if "replace_mode" not in st.session_state:
    st.session_state.replace_mode = False
if "replace_step" not in st.session_state:
    st.session_state.replace_step = 1  # 1: 选择替换源, 2: 选择替换目标
if "selected_replace_source" not in st.session_state:
    st.session_state.selected_replace_source = None
if "selected_replace_target" not in st.session_state:
    st.session_state.selected_replace_target = None
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
page = st.sidebar.radio("操作", ["查询产品", "批量查询", "产品选择", "添加产品", "删除产品"])
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
    # 新增英文关键词输入框
    keyword_en = st.text_input(
        "English Keyword (for searching Describrition_English)", key="keyword_en"
    )
    mat_kw = st.text_input(
        "物料号搜索", key="mat_kw"
    )
    
    # 价格级别选择框
    price_levels = [
        "全部显示",
        "二级代理A级别",
        "一级代理B级别", 
        "聚万大客户C级别",
        "青山大客户D级别",
        "大唐大客户E级别包运费"
    ]
    selected_price_level = st.selectbox(
        "选择价格级别",
        options=price_levels,
        key="price_level"
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
            
            # 根据选择的价格级别动态显示列
            base_cols = ["Material", "Describrition", "Describrition_English", "数量", "采购不含税"]
            exclude_cols = ["出厂价_含税", "出厂价_不含税", "NO"]
            
            # 价格级别映射（采购不含税始终显示，所以从映射中移除）
            price_level_mapping = {
                "二级代理A级别": ["二级代理A级别_利润率", "二级代理A级别_报单价格"],
                "一级代理B级别": ["一级代理B级别_利润率", "一级代理B级别_报单价格"],
                "聚万大客户C级别": ["聚万大客户C级别_利润率", "聚万大客户C级别_报单价格"],
                "青山大客户D级别": ["青山大客户D级别_利润率", "青山大客户D级别_报单价格"],
                "大唐大客户E级别包运费": ["大唐大客户E级别包运费_利润率", "大唐大客户E级别包运费_报单价格"]
            }
            
            if selected_price_level == "全部显示":
                # 显示除了出厂价和NO之外的所有列
                all_cols = [col for col in df.columns if col not in exclude_cols]
                show_cols = base_cols + [col for col in all_cols if col not in base_cols]
            else:
                # 显示基础信息 + 采购不含税 + 选择的价格级别对应的列
                selected_cols = price_level_mapping.get(selected_price_level, [])
                show_cols = base_cols + selected_cols

            # 优先物料号精确查找
            mat_kw = st.session_state.get("mat_kw", "").strip()
            if mat_kw:
                filtered = df[df["Material"].astype(str).str.contains(mat_kw)]
                if not filtered.empty:
                    out_df = pd.DataFrame(filtered.copy())  # 强制DataFrame
                    out_df["数量"] = qty
                    out_df["查询关键词"] = mat_kw  # 添加关键词列
                    out_df = out_df[[col for col in show_cols if col in out_df.columns] + ["查询关键词"]]
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
                        out_df["查询关键词"] = st.session_state.keyword  # 添加关键词列
                        show_cols_fuzzy = show_cols + ["匹配度", "查询关键词"]
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
                    out_df["查询关键词"] = st.session_state.keyword  # 添加关键词列
                    out_df = out_df[[col for col in show_cols if col in out_df.columns] + ["查询关键词"]]
                    st.session_state.last_out = out_df
                else:
                    st.session_state.last_out = pd.DataFrame()
                    st.warning("⚠️ 未查询到符合条件的产品")

                    # 新增英文查询按钮
        if st.button("查询英文描述", use_container_width=True):
            keyword_en = st.session_state.get("keyword_en", "").strip()
            qty = st.session_state.qty if "qty" in st.session_state else 1
            base_cols = ["Material", "Describrition", "Describrition_English", "数量", "采购不含税"]
            exclude_cols = ["出厂价_含税", "出厂价_不含税", "NO"]
            
            # 价格级别映射（采购不含税始终显示，所以从映射中移除）
            price_level_mapping = {
                "二级代理A级别": ["二级代理A级别_利润率", "二级代理A级别_报单价格"],
                "一级代理B级别": ["一级代理B级别_利润率", "一级代理B级别_报单价格"],
                "聚万大客户C级别": ["聚万大客户C级别_利润率", "聚万大客户C级别_报单价格"],
                "青山大客户D级别": ["青山大客户D级别_利润率", "青山大客户D级别_报单价格"],
                "大唐大客户E级别包运费": ["大唐大客户E级别包运费_利润率", "大唐大客户E级别包运费_报单价格"]
            }
            
            selected_price_level = st.session_state.get("price_level", "全部显示")
            if selected_price_level == "全部显示":
                # 显示除了出厂价和NO之外的所有列
                all_cols = [col for col in df.columns if col not in exclude_cols]
                show_cols = base_cols + [col for col in all_cols if col not in base_cols]
            else:
                # 显示基础信息 + 采购不含税 + 选择的价格级别对应的列
                selected_cols = price_level_mapping.get(selected_price_level, [])
                show_cols = base_cols + selected_cols

            if keyword_en:
                results_en = search_with_keywords(df, keyword_en, "Describrition_English", strict=True)
                if results_en:
                    out_df = pd.DataFrame(results_en)
                    out_df["数量"] = qty
                    out_df["查询关键词"] = keyword_en  # 添加英文关键词列
                    out_df = out_df[[col for col in show_cols if col in out_df.columns] + ["查询关键词"]]
                    st.session_state.last_out = out_df
                else:
                    st.session_state.last_out = pd.DataFrame()
                    st.warning("⚠️ 未查询到符合条件的英文产品")
            else:
                st.warning("请输入英文关键词")

    with query_c2:
        # The AI button is only active if there are fuzzy results to choose from
        can_ai_select = (
            "last_out" in st.session_state and
            not st.session_state.last_out.empty and
            "匹配度" in st.session_state.last_out.columns
        )
        if st.button("🤖 AI 优选", use_container_width=True, disabled=not can_ai_select):
            with st.spinner("🤖 AI 正在分析最佳匹配..."):
                top_5_df = st.session_state.last_out.head(5)
                if isinstance(top_5_df, pd.DataFrame):
                    best_choice_df, message = ai_select_best_with_gpt(
                        st.session_state.keyword, top_5_df
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
        out_df = prioritize_liansu(out_df)
        if "匹配度" in out_df.columns and "_liansu_priority" in out_df.columns:
            out_df = out_df.sort_values(["_liansu_priority", "匹配度"], ascending=[False, False]).reset_index(drop=True)
            out_df = out_df.drop("_liansu_priority", axis=1)
        elif "匹配度" in out_df.columns:
            out_df = out_df.sort_values("匹配度", ascending=False).reset_index(drop=True)
        else:
            out_df = out_df.reset_index(drop=True)
        
        # 重新排列列顺序，将查询关键词放在第一列
        if "查询关键词" in out_df.columns:
            cols = ["查询关键词"] + [col for col in out_df.columns if col != "查询关键词"]
            out_df = out_df[cols]
        
        # 重置索引，让序号从1开始
        out_df = out_df.reset_index(drop=True)
        out_df.index = out_df.index + 1
        
        st.dataframe(out_df, use_container_width=True)
        
        # 购物车操作区域
        st.subheader("购物车操作")
        
        # 选择要加入购物车的行
        to_cart = st.multiselect(
            "选择要加入购物车的行",
            options=list(out_df.index),
            format_func=lambda i: f"序号 {i}: {format_row(i, out_df)}",
            key="to_cart"
        )
        
        # 按钮布局
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("添加到购物车", key="add_cart"):
                for i in to_cart:
                    # 将1开始的索引转换为0开始的索引
                    actual_index = i - 1
                    st.session_state.cart.append(out_df.iloc[actual_index].to_dict())
                if "to_cart" in st.session_state:
                    del st.session_state["to_cart"]
                st.success("✅ 已加入购物车")
        
        with col2:
            # 替换功能
            if st.session_state.cart:
                if st.button("🔄 替换购物车项目", key="replace_cart_item"):
                    st.session_state.replace_mode = True
                    st.session_state.replace_step = 1
                    st.session_state.selected_replace_source = None
                    st.session_state.selected_replace_target = None
                    st.rerun()
            else:
                st.info("购物车为空，无法替换")

        # 替换弹窗
        if st.session_state.replace_mode:
            with st.container():
                st.markdown("---")
                st.subheader("🔄 替换购物车项目")
                
                if st.session_state.replace_step == 1:
                    # 第一步：选择要用来替换的查询结果
                    st.write("**步骤 1：选择要用来替换的产品**")
                    st.write("请从以下查询结果中选择一个产品：")
                    
                    # 确保out_df有连续的索引，并设置1基索引
                    out_df_for_replace = out_df.reset_index(drop=True)
                    out_df_for_replace.index = out_df_for_replace.index + 1
                    
                    # 显示查询结果表格
                    st.dataframe(out_df_for_replace, use_container_width=True)
                    
                    # 选择方式
                    selection_method = st.radio(
                        "请选择选择方式：",
                        ["通过序号选择", "通过下拉菜单选择"],
                        key="selection_method_step1"
                    )
                    
                    if selection_method == "通过序号选择":
                        # 序号输入方式
                        max_index = len(out_df_for_replace) if not out_df_for_replace.empty else 0
                        index_input = st.number_input(
                            f"请输入序号 (1-{max_index})",
                            min_value=1,
                            max_value=max_index,
                            value=1,
                            key="replace_source_index"
                        )
                        
                        # 显示选中序号对应的产品信息
                        if not out_df_for_replace.empty and 1 <= index_input <= len(out_df_for_replace):
                            selected_row = out_df_for_replace.iloc[index_input - 1]  # 转换为0基索引
                            st.info(f"当前选中序号 {index_input} 的产品：{selected_row.get('Describrition', 'N/A')}")
                            replace_source = index_input - 1  # 转换为0基索引用于后续操作
                        else:
                            st.warning("请输入有效的序号")
                            replace_source = None
                    
                    else:
                        # 下拉菜单方式
                        replace_source = st.selectbox(
                            "选择要用来替换的产品",
                            options=list(out_df_for_replace.index),
                            format_func=lambda i: f"序号 {i}: {format_row(i, out_df_for_replace)}",
                            key="replace_source_select"
                        )
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("继续", key="continue_to_step2"):
                            st.session_state.selected_replace_source = replace_source
                            st.session_state.replace_step = 2
                            st.rerun()
                    
                    with col2:
                        if st.button("取消", key="cancel_replace"):
                            st.session_state.replace_mode = False
                            st.session_state.replace_step = 1
                            st.session_state.selected_replace_source = None
                            st.session_state.selected_replace_target = None
                            st.rerun()
                
                elif st.session_state.replace_step == 2:
                    # 第二步：选择要替换的购物车项目
                    st.write("**步骤 2：选择要替换的购物车项目**")
                    st.write("请从购物车中选择要替换的项目：")
                    
                    # 重新获取out_df_for_replace，确保在第二步中也能访问
                    out_df_for_replace = out_df.reset_index(drop=True)
                    
                    # 显示购物车表格
                    cart_df = pd.DataFrame(st.session_state.cart)
                    if "查询关键词" in cart_df.columns:
                        cols = ["查询关键词"] + [col for col in cart_df.columns if col != "查询关键词"]
                        cart_df = cart_df[cols]
                    
                    # 重置索引，让序号从1开始
                    cart_df = cart_df.reset_index(drop=True)
                    cart_df.index = cart_df.index + 1
                    
                    st.dataframe(cart_df, use_container_width=True)
                    
                    # 选择方式
                    selection_method_step2 = st.radio(
                        "请选择选择方式：",
                        ["通过序号选择", "通过下拉菜单选择"],
                        key="selection_method_step2"
                    )
                    
                    if selection_method_step2 == "通过序号选择":
                        # 序号输入方式
                        max_cart_index = len(cart_df) if not cart_df.empty else 0
                        cart_index_input = st.number_input(
                            f"请输入购物车序号 (1-{max_cart_index})",
                            min_value=1,
                            max_value=max_cart_index,
                            value=1,
                            key="replace_target_index"
                        )
                        
                        # 显示选中序号对应的购物车项目信息
                        if not cart_df.empty and 1 <= cart_index_input <= len(cart_df):
                            selected_cart_row = cart_df.iloc[cart_index_input - 1]  # 转换为0基索引
                            st.info(f"当前选中序号 {cart_index_input} 的购物车项目：{selected_cart_row.get('Describrition', 'N/A')}")
                            replace_target = cart_index_input - 1  # 转换为0基索引用于后续操作
                        else:
                            st.warning("请输入有效的序号")
                            replace_target = None
                    
                    else:
                        # 下拉菜单方式
                        replace_target = st.selectbox(
                            "选择要替换的购物车项目",
                            options=list(cart_df.index),
                            format_func=lambda i: f"序号 {i}: {cart_df.loc[i, 'Describrition'] if 'Describrition' in cart_df.columns else cart_df.loc[i, '产品描述']}",
                            key="replace_target_select"
                        )
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("应用替换", key="apply_replace"):
                            # 添加调试信息
                            st.write(f"调试信息：")
                            st.write(f"- selected_replace_source: {st.session_state.selected_replace_source}")
                            st.write(f"- replace_target: {replace_target}")
                            st.write(f"- out_df_for_replace长度: {len(out_df_for_replace)}")
                            st.write(f"- cart长度: {len(st.session_state.cart)}")
                            
                            if st.session_state.selected_replace_source is not None and replace_target is not None:
                                try:
                                    # 执行替换 - 使用iloc而不是loc来避免索引问题
                                    new_item = out_df_for_replace.iloc[st.session_state.selected_replace_source].to_dict()
                                    st.session_state.cart[replace_target] = new_item
                                    st.success("✅ 替换成功！")
                                    # 重置状态
                                    st.session_state.replace_mode = False
                                    st.session_state.replace_step = 1
                                    st.session_state.selected_replace_source = None
                                    st.session_state.selected_replace_target = None
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"替换失败：{str(e)}")
                                    st.error(f"调试信息：selected_replace_source={st.session_state.selected_replace_source}, replace_target={replace_target}")
                                    st.error(f"out_df长度：{len(out_df)}, cart长度：{len(st.session_state.cart)}")
                    
                    with col2:
                        if st.button("返回", key="back_to_step1"):
                            st.session_state.replace_step = 1
                            st.session_state.selected_replace_source = None
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
                        top_3_df = candidates_df.head(3)
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
                        selected_item['查询关键词'] = keyword  # 添加查询关键词
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
                            "数量": quantity,
                            "查询关键词": keyword  # 添加查询关键词
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
            


elif page == "产品选择":
    st.header("🛒 产品选择")
    
    # 购物车展示和管理
    if st.session_state.cart:
        st.subheader("当前购物车")
        cart_df = pd.DataFrame(st.session_state.cart)
        
        # 重新排列列顺序，将查询关键词放在第一列（如果存在）
        if "查询关键词" in cart_df.columns:
            cols = ["查询关键词"] + [col for col in cart_df.columns if col != "查询关键词"]
            cart_df = cart_df[cols]
        
        # 重置索引，让序号从1开始
        cart_df = cart_df.reset_index(drop=True)
        cart_df.index = cart_df.index + 1
        
        # 显示购物车内容
        st.dataframe(cart_df, use_container_width=True)
        
        # 删除功能
        to_remove = st.multiselect(
            "选择要删除的购物车条目",
            options=list(cart_df.index),
            format_func=lambda i: f"序号 {i}: {cart_df.loc[i, 'Describrition'] if 'Describrition' in cart_df.columns else cart_df.loc[i, '产品描述']}",
            key="to_remove_cart"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("删除所选", key="del_cart_selection"):
                # 将1开始的索引转换为0开始的索引
                idxs = set([i - 1 for i in to_remove])
                st.session_state.cart = [it for j, it in enumerate(st.session_state.cart) if j not in idxs]
                if "to_remove_cart" in st.session_state:
                    del st.session_state["to_remove_cart"]
                st.rerun()
        
        with col2:
            if st.button("清空购物车", key="clear_cart"):
                st.session_state.cart = []
                st.rerun()
        
        # 显示购物车统计信息
        total_items = len(st.session_state.cart)
        st.info(f"购物车中共有 {total_items} 个产品")
        
    else:
        st.info("购物车为空，请先在\"查询产品\"或\"批量查询\"页面添加产品")

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
            # 跳过出厂价字段和NO字段
            if col in ["出厂价_含税","出厂价_不含税","NO"]:
                continue
            label = col + ("（必填）" if col in ["Describrition"] else "")
            dtype = df0[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                new_vals[col] = st.number_input(label, step=1, format="%d", key=f"add_{col}")
            elif pd.api.types.is_float_dtype(dtype):
                new_vals[col] = st.number_input(label, format="%.2f", key=f"add_{col}")
            else:
                new_vals[col] = st.text_input(label, key=f"add_{col}")

        submitted = st.form_submit_button("提交新增")

    if submitted:
        missing = [
            f for f in ["Describrition"]
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

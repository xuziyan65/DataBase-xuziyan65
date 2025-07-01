import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text
from pathlib import Path
from collections import Counter

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
    max-width: 1000px !important;
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
DB_PATH = Path(__file__).resolve().parents[1] / "Product1.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"timeout":20}, echo=False)

@st.cache_data
def load_data():
    return pd.read_sql("SELECT * FROM Products", engine)


def normalize_material(s: str) -> str:
    s = s.lower()
    s = s.replace('－', '-').replace('—', '-').replace('–', '-')
    s = re.sub(r'[\s_]', '', s)
    s = s.replace('（', '(').replace('）', ')')
    s = ''.join([chr(ord(c)-65248) if 65281 <= ord(c) <= 65374 else c for c in s])
    # 材质归一化
    s = re.sub(r'pp[\s\-_—–]?[rｒr]', 'ppr', s)  # 归一化pp-r、pp r、pp_r、pp—r、pp–r、ppｒ为ppr
    s = s.replace('pvcu', 'pvc')
    s = s.replace('pvc-u', 'pvc')
    s = s.replace('pvc u', 'pvc')
    # 只把常见分隔符替换成空格，保留*号
    s = re.sub(r'[\|,;，；]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    if "异径三通" in s:
        print("归一化后描述：", s)
    return s.strip()

def insert_product(values: dict):
    values.pop("序号", None)
    cols   = ", ".join(values.keys())
    params = ", ".join(f":{k}" for k in values)
    sql    = text(f"INSERT INTO Products ({cols}) VALUES ({params})")
    with engine.begin() as conn:
        conn.execute(sql, values)

def delete_products(materials: list[str]):
    if not materials:
        return
    with engine.begin() as conn:
        for m in materials:
            conn.execute(text("DELETE FROM Products WHERE Material = :m"), {"m": m})

# — 同义词 & 单位映射工具 —
SYNONYM_GROUPS = [
    {"直接", "直接头", "直通"},
    {"大小头", "异径直通", "异径套"},
    {"扫除口", "清扫口"}
]

mm_to_inch = {"20": '1/2"', "25": '3/4"',
              "32": '1"', "50": '1-1/2"',
              "63": '2"', "75": '2-1/2"',
              "90": '3"', "110": '4"'}
inch_to_mm = {v:k for k,v in mm_to_inch.items()}

def get_synonym_words(word):
    for group in SYNONYM_GROUPS:
        if word in group:
            return group
    return {word}

def expand_unit_tokens(token):
    eqs = {token}
    # 支持dn前缀
    if token.startswith('dn'):
        num = token[2:]
        if num in mm_to_inch:
            eqs.add(mm_to_inch[num])
            # 也可以加 'dn'+英寸
            eqs.add('dn' + mm_to_inch[num])
            if num in inch_to_mm:
                eqs.add('dn' + inch_to_mm[num])
    else:
        if token in mm_to_inch:
            eqs.add(mm_to_inch[token])
        if token in inch_to_mm:
            eqs.add(inch_to_mm[token])
    return eqs

def expand_token_with_synonyms_and_units(token):
    # 先查同义词组
    synonyms = get_synonym_words(token)
    expanded = set()
    for syn in synonyms:
        expanded |= expand_unit_tokens(syn)
    return expanded
    
def split_with_synonyms(text):
    text = text.lower()
    tokens = []
    # 先提取 dn+数字*数字
    pattern_dn = re.compile(r'dn\d+\*\d+')
    for m in pattern_dn.finditer(text):
        tokens.append(m.group())
    text = pattern_dn.sub(' ', text)
    # 再提取 数字*数字
    pattern_num = re.compile(r'\d+\*\d+')
    for m in pattern_num.finditer(text):
        tokens.append(m.group())
    text = pattern_num.sub(' ', text)
    # 再提取连续英文/拼音
    pattern_en = re.compile(r'[a-zA-Z]+')
    for m in pattern_en.finditer(text):
        tokens.append(m.group())
    text = pattern_en.sub(' ', text)
    # 再提取单个数字
    pattern_digit = re.compile(r'\d+')
    for m in pattern_digit.finditer(text):
        tokens.append(m.group())
    text = pattern_digit.sub(' ', text)
    # 剩下的按单字切分
    tokens += [c for c in text if c.strip()]
    return tokens

def classify_tokens(keyword):
    norm_kw = normalize_material(keyword)
    # 材质
    material_tokens = re.findall(r'pvc|ppr|pe|pp|hdpe|pb|pert', norm_kw)
    # 数字
    digit_tokens = re.findall(r'\d', norm_kw)
    # 中文同义词整体切分
    chinese_tokens = split_with_synonyms(keyword)
    return material_tokens, digit_tokens, chinese_tokens

def search_with_keywords(df, keyword, field, strict=True, return_score=False):
    material_tokens, digit_tokens, chinese_tokens = classify_tokens(keyword.strip())
    digit_counter = Counter(digit_tokens)
    results = []
    for row in df.itertuples(index=False):
        text = normalize_material(str(getattr(row, field, "")))
        # 材质必须全部命中
        if not all(m in text for m in material_tokens):
            continue
        text_digit_counter = Counter(re.findall(r'\d', text))
        if not strict:
            # 模糊查询时，要求数字出现和次数与输入一致
            if digit_counter and text_digit_counter != digit_counter:
                continue
        # 中文部分（同义词扩展）
        hit_count = 0
        if strict:
            if not all(any(syn in text for syn in expand_token_with_synonyms_and_units(c)) for c in chinese_tokens):
                continue
            hit_count = len(chinese_tokens)
        else:
            hit_count = sum(1 for c in chinese_tokens if any(syn in text for syn in expand_token_with_synonyms_and_units(c)))
            if chinese_tokens and hit_count == 0:
                continue
        if return_score:
            score = hit_count / len(chinese_tokens) if chinese_tokens else 1
            results.append((row, score))
        else:
            results.append(row)
    return results

# — Session State 初始化 —
for k, default in [("cart",[]),("last_out",pd.DataFrame()),("to_cart",[]),("to_remove",[])]:
    if k not in st.session_state:
        st.session_state[k] = default

def add_to_cart():
    for i in st.session_state.to_cart:
        st.session_state.cart.append(st.session_state.last_out.loc[i].to_dict())
    # 清空选择（推荐用 pop 或 del）
    if "to_cart" in st.session_state:
        del st.session_state["to_cart"]

def remove_from_cart():
    idxs = set(st.session_state.to_remove)
    st.session_state.cart = [it for j,it in enumerate(st.session_state.cart) if j not in idxs]
    if "to_remove" in st.session_state:
        del st.session_state["to_remove"]

# — 侧边栏导航 —
st.sidebar.header("导  航")
page = st.sidebar.radio("操作", ["查询产品","添加产品","删除产品"])
st.sidebar.markdown("---")
st.sidebar.caption("Powered by Streamlit")


# 页面切换和主逻辑
if page == "查询产品":
    st.header("产品查询系统")
    df = load_data()

    
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

    if st.button("查询"):
        out_df = pd.DataFrame()
        qty = st.session_state.qty if "qty" in st.session_state else 1
        show_cols = ["Material", "Describrition", "数量", "出厂价_含税", "出厂价_不含税"]

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
            if not results and st.session_state.fuzzy_mode:
                fuzzy_results = search_with_keywords(df, st.session_state.keyword, "Describrition", strict=False, return_score=True)
                if fuzzy_results:
                    out_df = pd.DataFrame([r[0] for r in fuzzy_results])
                    out_df["匹配度"] = [round(r[1], 2) for r in fuzzy_results]
                    out_df = out_df.sort_values("匹配度", ascending=False)
                    out_df["数量"] = qty
                    show_cols_fuzzy = show_cols + ["匹配度"]
                    out_df = out_df[[col for col in show_cols_fuzzy if col in out_df.columns]]
                    st.session_state.last_out = out_df
                else:
                    st.session_state.last_out = pd.DataFrame()
                    st.warning("⚠️ 未查询到符合条件的产品")
            elif results:
                out_df = pd.DataFrame(results)
                out_df["数量"] = qty
                out_df = out_df[[col for col in show_cols if col in out_df.columns]]
                st.session_state.last_out = out_df
            else:
                st.session_state.last_out = pd.DataFrame()
                st.warning("⚠️ 未查询到符合条件的产品")

    # 查询结果展示和购物车操作（无论是否刚点了查询按钮，只要有结果都显示）
    out_df = st.session_state.get("last_out", pd.DataFrame())
    if not out_df.empty and isinstance(out_df, pd.DataFrame):
        st.dataframe(out_df, use_container_width=True)
        def format_row(i):
            try:
                row = out_df.loc[i]
                if "产品描述" in out_df.columns:
                    return row["产品描述"]
                elif "Describrition" in out_df.columns:
                    return row["Describrition"]
                elif "Material" in out_df.columns:
                    return str(row["Material"])
                else:
                    return str(i)
            except Exception:
                return str(i)
        to_cart = st.multiselect(
            "选择要加入购物车的行",
            options=list(out_df.index),
            format_func=format_row,
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

elif page == "添加产品":
    st.header(" 添加新产品到数据库")
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



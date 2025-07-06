import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from pathlib import Path
from collections import Counter
from openai import OpenAI

# --- 安全地从 Streamlit Secrets 获取 API KEY ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

# --- AI 选择功能 (GPT-4o-mini) ---
def ai_select_best_with_gpt(keyword: str, df: pd.DataFrame):
    """
    Uses GPT-4o-mini to select the best match from a DataFrame of candidates.
    """
    if not OPENAI_API_KEY:
        return None, "错误：请在 Streamlit Cloud 的 Secrets 中设置您的 OpenAI API Key。"

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Create a string representation of the choices for the prompt
    choices_str = ""
    # Use a fresh, reset index for this operation to guarantee alignment
    df_reset = df.reset_index(drop=True)
    for i, row in df_reset.iterrows():
        choices_str += f"索引 {i}: {row['Describrition']}\n" # 使用真实的换行符

    prompt_lines = [
        "你是一个专业的管道建材产品采购专家。你的任务是从一个产品列表中，根据用户的搜索请求，选出最匹配的一项。",
        f"用户的搜索请求是: \"{keyword}\"",
        f"以下是系统模糊匹配出的最相关的{len(df_reset)}个候选产品:",
        "---",
        choices_str,
        "---",
        "请仔细分析用户的请求和每个候选产品的描述，选出最符合用户意图的**唯一一个**产品。",
        "你的回答必须严格遵循以下格式，只返回你选择的产品的**索引数字**，不要添加任何其他内容。",
        "例如:",
        "2"
    ]
    prompt = "\n".join(prompt_lines)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "你是一个专业的管道建材产品采购专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10, # Only need a number
            timeout=20,
        )
        
        content = response.choices[0].message.content.strip()
        
        if not content.isdigit():
             raise ValueError(f"AI未能返回有效的索引数字。原始回复: '{content}'")

        selected_index = int(content)

        if selected_index < 0 or selected_index >= len(df_reset):
             raise ValueError(f"AI返回了越界的索引: {selected_index}。")

        # Return the single selected row (using the correct index)
        best_row_df = df_reset.iloc[[selected_index]]
        return best_row_df, "Success"

    except Exception as e:
        error_message = str(e)
        if "Incorrect API key" in error_message:
            return None, "AI调用失败：API Key不正确或已失效。请检查 Streamlit Cloud 中的配置。"
        return None, f"AI调用失败：{error_message}"
# --- 结束 AI 功能 ---

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
@st.cache_resource
def get_db_engine():
    """
    Creates a cached database engine for the Streamlit app.
    Using @st.cache_resource ensures that the connection is established only once
    per session. The StaticPool is crucial for SQLite to prevent "database is locked"
    errors in Streamlit's multi-threaded environment by ensuring all operations
    use the same underlying connection.
    """
    DB_PATH = Path(__file__).resolve().parents[1] / "Product2.db"
    engine = create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    return engine

engine = get_db_engine()


#从数据库读取产品数据，并对结果进行缓存
@st.cache_data
def load_data():
    return pd.read_sql("SELECT * FROM Products", engine)
if "show_more" not in st.session_state:
    st.session_state.show_more = False
    
def is_token_in_text(token, text):
    # 匹配完整的英寸单位
    # 前面不是数字或斜杠，后面不是数字、斜杠或连字符
    return re.search(rf'(?<![\d/-]){re.escape(token)}(?![\d/\\-])', text) is not None
# 归一化产品描述，将常见变体统一为标准形式
def normalize_material(s: str) -> str:
    s = s.lower()
    s = s.replace('－', '-').replace('—', '-').replace('–', '-')
    s = re.sub(r'[_\t]', ' ', s)
    s = s.replace('（', '(').replace('）', ')')
    s = s.replace('x', '*') # 统一尺寸分隔符
    s = ''.join([chr(ord(c)-65248) if 65281 <= ord(c) <= 65374 else c for c in s])
    # 材质归一化
    s = re.sub(r'pp[\s\-_—–]?[rｒr]', 'ppr', s)  # 归一化pp-r、pp r、pp_r、pp—r、pp–r、ppｒ为ppr
    s = s.replace('pvcu', 'pvc')
    s = s.replace('pvc-u', 'pvc')
    s = s.replace('pvc u', 'pvc')
    # 只把常见分隔符替换成空格，保留*号
    s = re.sub(r'[\|,;，；、]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    # 统一英寸符号
    s = s.replace('＂', '"').replace('"', '"').replace('"', '"')
    s = re.sub(r'\\s*\"\\s*', '"', s)  # 去除英寸符号前后空格
    s = s.replace('in', '"')           # 2in -> 2"
    s = s.replace('英寸', '"')
    s = s.replace('寸', '"')
    # 可根据实际情况添加更多变体
    return s.strip()

# 插入新产品的函数
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
    {"扫除口", "清扫口", "检查口"},
    {"内丝", "内螺纹"},
    {"双联", "双联座"}
]

# PVC管道英寸-毫米对照
mm_to_inch_pvc = {
    "16": '1/2"', "20": '3/4"', "25": '1"', "35": '1-1/4"', "40": '1-1/2"', "50": '2"',
    "65": '2-1/2"', "75": '3"', "100": '4"', "125": '5"', "150": '6"', "200": '8"',
    "250": '10"', "300": '12"'
}
inch_to_mm_pvc = {v: k for k, v in mm_to_inch_pvc.items()}

# PPR管道英寸-毫米对照
mm_to_inch_ppr = {
    "20": '1/2"', "25": '3/4"', "32": '1"', "40": '1-1/4"', "50": '1-1/2"', "63": '2"',
    "75": '2-1/2"', "90": '3"', "110": '4"', "160": '6"'
}
inch_to_mm_ppr = {v: k for k, v in mm_to_inch_ppr.items()}

#查找某个词的同义词集合，用于后续检索时自动扩展同义词匹配。如果没有同义词，就只返回自己。
def get_synonym_words(word):
    for group in SYNONYM_GROUPS:
        if word in group:
            return group
    return {word}

# 扩展单位符号，比如dn20*20，会扩展为dn20、dn20*20、20、20*20
def expand_unit_tokens(token, material=None):
    eqs = {token}
    # 选择对照表
    if material == "pvc":
        mm_to_inch = mm_to_inch_pvc
        inch_to_mm = inch_to_mm_pvc
    elif material == "ppr":
        mm_to_inch = mm_to_inch_ppr
        inch_to_mm = inch_to_mm_ppr
    else:
        mm_to_inch = {**mm_to_inch_pvc, **mm_to_inch_ppr}
        inch_to_mm = {**inch_to_mm_pvc, **inch_to_mm_ppr}
    
    # Case 0: Handle composite specs like "20*1/2""
    m = re.fullmatch(r'(?:dn)?(\d+)\*(.+)', token)
    if m:
        part1_mm = m.group(1)
        part2_inch_str = m.group(2)
        if part1_mm in mm_to_inch:
            eqs.add(f"dn{part1_mm}*{part2_inch_str}")
            eqs.add(f"{part1_mm}*{part2_inch_str}")
        return eqs

    # Case 1: 'dn' value, e.g., 'dn25'
    if token.startswith('dn'):
        num = token[2:]
        if num in mm_to_inch:
            eqs.add(mm_to_inch[num]) # '3/4"'
        eqs.add(num) # '25'
        return eqs

    # Case 2: An inch value, quoted or not, e.g., '3/4"' or '3/4'
    inch_lookup_token = token
    # Add quote if it's a fraction like "1/2", "1-1/2"
    if re.fullmatch(r'\d+-\d+/\d+|\d+/\d+', token):
        inch_lookup_token = token + '"' # '3/4' -> '3/4"'
    
    if inch_lookup_token in inch_to_mm:
        mm_val = inch_to_mm[inch_lookup_token] # '25'
        eqs.add(mm_val)
        eqs.add('dn' + mm_val) # 'dn25'
        eqs.add(inch_lookup_token) # '3/4"'
        return eqs

    # Case 3: A plain number, could be mm, e.g., '25'
    if token.isdigit() and token in mm_to_inch:
        eqs.add('dn' + token) # 'dn25'
        eqs.add(mm_to_inch[token]) # '3/4"'
        return eqs

    return eqs


#前两个函数的集合
def expand_token_with_synonyms_and_units(token, material=None):
    # 先查同义词组
    synonyms = get_synonym_words(token)
    expanded = set()
    for syn in synonyms:
        expanded |= expand_unit_tokens(syn, material=material)
    return expanded

# 将中文描述切分为单词列表，并自动扩展同义词和单位符号，
# "PPR dn20*25 直接头"，['ppr', 'dn20*25', '20*25', '20', '25', '直接', '头']
def split_with_synonyms(text):
    # 0. 预处理：标准化各种可能造成解析问题的字符
    text = text.replace('（', '(').replace('）', ')')
    text = text.replace('＂', '"')  # 全角引号
    text = text.replace('－', '-')  # 全角连字符

    # 预分词：解决 "PPRDN20" 这类连写问题
    text = re.sub(r'([A-Z])(DN)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'(DN)(\d)', r'\1 \2', text, flags=re.IGNORECASE)

    # 移除括号，防止干扰英寸规格解析
    text = text.replace('(', ' ').replace(')', ' ')

    text = text.lower()
    text = text.replace('*', ' * ')
    # 统一顿号、逗号等
    text = text.replace('、', ' ').replace('，', ' ').replace('；', ' ')

    # 新增：统一数字和英寸单位的组合
    text = re.sub(r'(\d+(?:\.\d+)?)\s*(in|寸|英寸)', r'\1"', text)

    tokens = []

    # NEW: 优先提取键值对, 如 "pn=1.0", "pn:10"
    # 这样可以正确地将键和值分开，并且能处理浮点数
    pattern_kv = re.compile(r'([a-zA-Z]+)\s*[:=]\s*(\d+(?:\.\d+)?)')
    for m in pattern_kv.finditer(text):
        tokens.append(m.group(1))  # key, e.g., "pn"
        tokens.append(m.group(2))  # value, e.g., "1.0"
    text = pattern_kv.sub(' ', text)

    # NEW: Handle mixed mm*inch specs first
    # e.g., dn20*1/2"
    pattern_dn_mixed = re.compile(r'dn(\d+)\*(\d+/\d+"|\d+")')
    for m in pattern_dn_mixed.finditer(text):
        tokens.append(m.group(0)) # dn20*1/2"
        tokens.append(m.group(1)) # 20
        tokens.append(m.group(2)) # 1/2"
    text = pattern_dn_mixed.sub(' ', text)

    # e.g., 20*1/2"
    pattern_mixed = re.compile(r'(\d+)\*(\d+/\d+"|\d+")')
    for m in pattern_mixed.finditer(text):
        tokens.append(m.group(0)) # 20*1/2"
        tokens.append(m.group(1)) # 20
        tokens.append(m.group(2)) # 1/2"
    text = pattern_mixed.sub(' ', text)
    
    # 新增：处理角度规格，如 90°
    pattern_angle = re.compile(r'\d+°')
    for m in pattern_angle.finditer(text):
        tokens.append(m.group(0))
    text = pattern_angle.sub(' ', text)

    # 先提取 dn+数字*数字
    pattern_dn_star = re.compile(r'dn(\d+)\*(\d+)')
    for m in pattern_dn_star.finditer(text):
        tokens.append(m.group())
        tokens.append(f"{m.group(1)}*{m.group(2)}")
        tokens.append(m.group(1))
        tokens.append(m.group(2))
    text = pattern_dn_star.sub(' ', text)
    # 再提取 dn+数字
    pattern_dn = re.compile(r'dn(\d+)')
    for m in pattern_dn.finditer(text):
        tokens.append(m.group())
        tokens.append(m.group(1))
    text = pattern_dn.sub(' ', text)
    # 再提取 数字*数字
    pattern_num = re.compile(r'(\d+)\*(\d+)')
    for m in pattern_num.finditer(text):
        tokens.append(m.group())
        tokens.append(m.group(1))
        tokens.append(m.group(2))
    text = pattern_num.sub(' ', text)
    
    # 修正：更新英寸正则表达式以支持小数点
    pattern_inch = re.compile(r'\d+-\d+/\d+"|\d+/\d+"|(?:\d+\.\d+|\d+)"')
    for m in pattern_inch.finditer(text):
        tokens.append(m.group())
    text = pattern_inch.sub(' ', text)

    # 新增: 提取不带引号的分数 (e.g. 3/4, 1-1/2)
    pattern_fraction = re.compile(r'\d+-\d+/\d+|\d+/\d+')
    for m in pattern_fraction.finditer(text):
        tokens.append(m.group())
    text = pattern_fraction.sub(' ', text)

    # 再提取连续英文/拼音
    pattern_en = re.compile(r'[a-zA-Z]+')
    for m in pattern_en.finditer(text):
        tokens.append(m.group())
    text = pattern_en.sub(' ', text)
    # 再提取单个数字 (包括小数)
    pattern_digit = re.compile(r'\d+(?:\.\d+)?')
    for m in pattern_digit.finditer(text):
        tokens.append(m.group())
    text = pattern_digit.sub(' ', text)
    # 剩下的按单字切分
    tokens += [c for c in text if c.strip()]
    
    # 去掉 'dnXX'，如果 'XX' 也在 tokens 里
    filtered = []
    token_set = set(tokens)
    for t in tokens:
        if re.fullmatch(r'dn(\d+)', t):
            num = t[2:]
            if num in token_set:
                continue  # 跳过 'dnXX'
        filtered.append(t)
    return filtered

#前三函数总和，输入："PPR dn20*25 直接头"
#输出：material_tokens: ['ppr']
#digit_tokens: ['2', '0', '2', '5']
#chinese_tokens: ['ppr', 'dn20*25', '20*25', '20', '25', '直接', '头']
def classify_tokens(keyword):
    norm_kw = normalize_material(keyword)
    # 材质
    material_tokens = re.findall(r'pvc|ppr|pe|pp|hdpe|pb|pert', norm_kw)
    # 数字 (修正：匹配包括小数在内的完整数字)
    digit_tokens = re.findall(r'\d+(?:\.\d+)?', norm_kw)
    # 中文同义词整体切分
    chinese_tokens = split_with_synonyms(keyword)
    return material_tokens, digit_tokens, chinese_tokens


def expand_keyword_with_synonyms(keyword: str) -> list[str]:
    """
    Expands a keyword into a list of queries with synonyms replaced.
    This happens BEFORE tokenization.
    Example: "直接dn20" -> ["直接dn20", "直接头dn20", "直通dn20"]
    """
    # Create a map from a synonym to its group for easy lookup.
    synonym_to_group = {syn: group for group in SYNONYM_GROUPS for syn in group}
    # Sort all available synonyms by length, descending.
    # This ensures that longer synonyms (e.g., "异径直通") are matched before
    # their shorter substrings (e.g., "直通").
    sorted_synonyms = sorted(synonym_to_group.keys(), key=len, reverse=True)

    # Use a set for the initial list of queries to handle duplicates.
    queries = {keyword}

    # Iterate through the sorted list of synonyms.
    for syn in sorted_synonyms:
        # Create a temporary list to hold newly generated queries.
        new_queries = set()
        for q in queries:
            # If the synonym is found in the current query...
            if syn in q:
                # ...get its synonym group.
                group = synonym_to_group[syn]
                # And for each synonym in that group...
                for replacement in group:
                    # ...create a new query by replacing the original synonym.
                    new_queries.add(q.replace(syn, replacement))
        
        # After checking all existing queries for a given synonym,
        # update the main set of queries with the new variations.
        # This prevents replacement loops and combinatorial explosion within a single pass.
        if new_queries:
            queries.update(new_queries)

    return list(queries)


def search_with_keywords(df, keyword, field, strict=True, return_score=False):
    # --- MODIFIED: Expand keyword with synonyms before processing ---
    expanded_keywords = expand_keyword_with_synonyms(keyword.strip())
    all_results = {} # Use dict to store unique results with the best score

    for kw in expanded_keywords:
        material_tokens, _, chinese_tokens = classify_tokens(kw)

        query_size_tokens = {t for t in chinese_tokens if re.search(r'\d', t) and not t.endswith('°')}
        query_text_tokens = {t for t in chinese_tokens if not (re.search(r'\d', t) and not t.endswith('°'))}

        # 1. 为每个查询规格，创建一个包含所有等价写法的集合
        query_spec_equivalents = {}
        query_material = material_tokens[0] if material_tokens else None
        for token in query_size_tokens:
            query_spec_equivalents[token] = expand_token_with_synonyms_and_units(token, material=query_material)
        
        for row in df.itertuples(index=False):
            # Use a unique identifier for each row to handle duplicates
            row_identifier = getattr(row, "Describrition", str(row)) 
            raw_text = str(getattr(row, field, ""))
            normalized_text = normalize_material(raw_text)

            if not all(m in normalized_text for m in material_tokens):
                continue

            product_all_tokens = split_with_synonyms(raw_text)
            text_specs = {t for t in product_all_tokens if re.search(r'\d', t)}
            
            if len(query_size_tokens) > len(text_specs):
                continue
    
            if query_size_tokens:
                unmatched_text_specs = text_specs.copy()
                all_query_specs_matched = True
                for q_spec, q_equivalents in query_spec_equivalents.items():
                    match_found = False
                    for t_spec in list(unmatched_text_specs):
                        if t_spec in q_equivalents:
                            match_found = True
                            unmatched_text_specs.remove(t_spec)
                            break
                    if not match_found:
                        all_query_specs_matched = False
                        break
                
                if not all_query_specs_matched:
                    continue

            material_keywords_in_query = {'pvc', 'ppr'}.intersection(query_text_tokens)
            if material_keywords_in_query:
                if not any(mat in normalized_text.lower() for mat in material_keywords_in_query):
                    continue

            hit_count = len(query_size_tokens)
            
            if strict:
                if not all(t in normalized_text.lower() for t in query_text_tokens):
                    continue
                hit_count += len(query_text_tokens)
            else:
                product_text_lower = normalized_text.lower()
                for token in query_text_tokens:
                    if token in product_text_lower:
                        hit_count += 1
                if query_text_tokens and hit_count == len(query_size_tokens):
                    continue
            
            total_tokens = len(query_size_tokens) + len(query_text_tokens)
            score = hit_count / total_tokens if total_tokens > 0 else 1
            
            # Store or update the result if the score is higher
            if row_identifier not in all_results or score > all_results[row_identifier][1]:
                all_results[row_identifier] = (row, score)

    # Convert the results dict back to a list
    final_results = list(all_results.values())
    # Sort results by score, descending
    final_results.sort(key=lambda x: x[1], reverse=True)

    if return_score:
        return final_results
    else:
        return [res[0] for res in final_results]

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
            base_cols = ["Material", "Describrition", "数量"]
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
                best_choice_df, message = ai_select_best_with_gpt(
                    st.session_state.keyword, top_3_df
                )
            
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

elif page == "批量查询":
    st.header("📦 批量导入查询")
    st.info("请上传一个 Excel (.xlsx) 或 CSV (.csv) 文件。文件中需要包含 **名称**、**规格** 和 **数量** 列。")

    uploaded_file = st.file_uploader(
        "上传查询文件",
        type=["xlsx", "csv"],
        key="batch_file_uploader"
    )

    if uploaded_file is not None:
        # 为了避免在每次交互时都重新读取文件，我们将其存储在会话状态中
        # 并检查上传的文件是否是新的
        if 'query_df' not in st.session_state or st.session_state.get('uploaded_filename') != uploaded_file.name:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.query_df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.query_df = pd.read_excel(uploaded_file)
                st.session_state.uploaded_filename = uploaded_file.name
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
            # 数据帧已加载，列名已选择。我们可以直接开始处理。
            products_df = load_data()
            results_log = []
            
            progress_bar = st.progress(0, text="正在准备批量查询...")
            total_rows = len(query_df)
            
            with st.spinner("正在逐条查询并使用 AI 优选，请稍候..."):
                for index, row in query_df.iterrows():
                    progress_text = f"正在处理: {index + 1}/{total_rows}"
                    progress_bar.progress((index + 1) / total_rows, text=progress_text)
                    
                    # Combine name and spec, then clean it
                    name_val = str(row[name_col]) if pd.notna(row[name_col]) else ""
                    spec_val = str(row[spec_col]) if pd.notna(row[spec_col]) else ""
                    
                    # 关键修正：直接合并，不再进行独立的标点清理。
                    # 所有的清理和解析都统一由 search_with_keywords 函数处理，以保证逻辑一致。
                    keyword = f"{name_val} {spec_val}".strip()
                    
                    # Ensure quantity is a valid number, default to 1 if not
                    try:
                        quantity = int(row.get(quantity_col, 1))
                    except (ValueError, TypeError):
                        quantity = 1


                    best_choice_df = None
                    status = "未找到"

                    # Step 1: Strict search
                    strict_results = search_with_keywords(products_df, keyword, "Describrition", strict=True)
                    
                    if strict_results:
                        candidates_df = pd.DataFrame(strict_results)
                        # Use AI to select from strict results (take top 5 to be safe with token limits)
                        best_choice_df, message = ai_select_best_with_gpt(keyword, candidates_df.head(5))
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
                            best_choice_df, message = ai_select_best_with_gpt(keyword, fuzzy_df.head(3))
                            if message == "Success" and best_choice_df is not None and not best_choice_df.empty:
                                status = "🟡 AI从模糊匹配结果中选择"

                    # Step 3: Add to cart if AI made a selection
                    if best_choice_df is not None and not best_choice_df.empty:
                        selected_item = best_choice_df.iloc[0].to_dict()
                        selected_item['数量'] = quantity
                        st.session_state.cart.append(selected_item)
                        results_log.append({
                            "查询关键词": keyword,
                            "匹配结果": selected_item.get("Describrition", "N/A"),
                            "状态": status
                        })
                    else:
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
                st.dataframe(pd.DataFrame(st.session_state.cart), use_container_width=True)

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





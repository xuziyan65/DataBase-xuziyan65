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

# — 同义词 & 单位映射工具 —
SYNONYM_GROUPS = [
    {"直接", "直接头", "直通","直通接头"},
    {"变径","异径"},
    {"大小头", "异径直通", "异径套","变径直接","异径直接","异径直通接头"},
    {"PN10", "1.0MPa"},
    {"PN16", "1.6MPa"},
    {"扫除口", "清扫口", "检查口","立管检查口"},
    {"内丝", "内螺纹"},
    {"外丝", "外螺纹"},
    {"锁母","锁扣","锁母锁扣","管接头"},
    {"线管直接","管直通"},
    {"止回阀","截止阀"},
    {"穿线管","电线管"},
    {"承插","承插式","对接"}
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

@st.cache_resource
def get_db_engine():
    DB_PATH = Path(__file__).resolve().parents[1] / "Product2.db"
    engine = create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    return engine

engine = get_db_engine()

@st.cache_data
def load_data():
    return pd.read_sql("SELECT * FROM Products", engine)
if "show_more" not in st.session_state:
    st.session_state.show_more = False

def ai_select_best_with_gpt(keyword: str, df: pd.DataFrame):
    """
    Uses GPT-4o-mini to select the best match from a DataFrame of candidates.
    """
    if not OPENAI_API_KEY:
        return None, "错误：请在 Streamlit Cloud 的 Secrets 中设置您的 OpenAI API Key。"

    client = OpenAI(api_key=OPENAI_API_KEY)

    #把所有候选产品的描述拼成一段文本，带上索引号
    choices_str = ""
    # Use a fresh, reset index for this operation to guarantee alignment
    df_reset = df.reset_index(drop=True)
    for i, row in df_reset.iterrows():
        choices_str += f"索引 {i}: {row['Describrition']}\n" # 使用真实的换行符

    prompt_lines = [
        "你是一个专业的管道建材产品采购专家。",
        "你的任务是从一个产品列表中，根据用户的搜索请求，选出最匹配的一项。",
        "**请严格遵守以下规则：**",
        "1. 如果用户的搜索请求中包含\"异径直接\"，你必须优先选择描述为\"异径套\"的产品。如果包含90°，优先选描述为90°的产品",
        "2. 如果用户的搜索请求中包含“PVC排水管”，你必须优先选择描述为“扩直口管”的产品，而不是“直通”或“管件”。",
        "3. 如果用户的搜索请求中包含“PVC给水管”，你必须优先选择描述为“印尼(日标)PVC-U给水扩直口管”的产品，而不是“直通”或“管件”。",
        "4. 你的回答必须只包含你选择的产品的**索引数字**，不要有任何其他文字、解释或标点符号。",
        "---",
        f"用户的搜索请求是: \"{keyword}\"",
        f"以下是候选产品列表 (共{len(df_reset)}个):",
        choices_str,
        "---",
        "请根据以上规则和用户请求，从列表中选出最匹配的**唯一一个**产品，并仅返回其索引号。"
    ]
    prompt = "\n".join(prompt_lines)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个专业的管道建材产品采购专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10, # Only need a number
            timeout=20,
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("AI返回了空内容")
        content = content.strip() #取出AI回复的数字，去掉首尾空格
        
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

def is_token_in_text(token, text):
    # 匹配完整的英寸单位
    # 前面不是数字或斜杠，后面不是数字、斜杠或连字符
    return re.search(rf'(?<![\d/-]){re.escape(token)}(?![\d/\\-])', text) is not None

# 归一化产品描述，将常见变体统一为标准形式
def normalize_material(s: str) -> str:
    s = s.lower() #转成小写
    s = s.replace('－', '-').replace('—', '-').replace('–', '-') #全角转半角
    s = re.sub(r'[_\t]', ' ', s)
    s = s.replace('（', '(').replace('）', ')')
    s = s.replace('x', '*') # 统一尺寸分隔符
    # 新增：将“90度”替换为“90°”
    s = s.replace('90度', '90°')
    # --- NEW: Handle various diameter symbols ---
    s = s.replace('ф', ' ').replace('φ', ' ').replace('ø', ' ').replace('⌀', ' ')
    s = ''.join([chr(ord(c)-65248) if 65281 <= ord(c) <= 65374 else c for c in s])

    # 材质归一化
    # 只归一化ppr相关，其他材质严格区分
    s = re.sub(r'pp[\s\-_—–]?[rｒr]', 'ppr', s)  # 归一化pp-r、pp r、pp_r、pp—r、pp–r、ppｒ为ppr
    # 不再将pvcu、pvc-u、pvc u归一化为pvc，保持原样
    # 只把常见分隔符替换成空格，保留*号
    s = re.sub(r'[\|,;，；、]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    # 统一英寸符号
    s = s.replace('“', '"').replace('”', '"')  # 处理中文书名号
    s = s.replace('＂', '"')                   # 全角引号
    s = s.replace('in', '"')                   # in → "
    s = s.replace('英寸', '"')
    s = s.replace('寸', '"')
    s = re.sub(r'\s*"\s*', '"', s)  # 去除英寸符号前后空格
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

#查找某个词的同义词集合，用于后续检索时自动扩展同义词匹配。如果没有同义词，就只返回自己。
def get_synonym_words(word):
    for group in SYNONYM_GROUPS:
        if word in group:
            return group
    return {word}

# 扩展单位符号，比如dn20*20，会扩展为dn20*20、20*20、20、20*20
def expand_unit_tokens(token, material=None):
    eqs = {token}
    # 新增：处理1.2"和1.5"的特殊映射
    special_inch_map = {
        '1.2"': '1-1/4"',
        '1.5"': '1-1/2"',
    }
    if token in special_inch_map:
        eqs.add(special_inch_map[token])
        return eqs
    # 选择对照表
    if material and material.startswith("pvc"):
        mm_to_inch = mm_to_inch_pvc
        inch_to_mm = inch_to_mm_pvc
    elif material == "ppr":
        mm_to_inch = mm_to_inch_ppr
        inch_to_mm = inch_to_mm_ppr
    else:
        # 默认用pvc
        mm_to_inch = mm_to_inch_pvc
        inch_to_mm = inch_to_mm_pvc
    
    # Case 0: Handle composite specs like "20*1/2""，扩展为dn20*1/2"和20*1/2"
    m = re.fullmatch(r'(?:dn)?(\d+)\*(.+)', token)
    if m:
        part1_mm = m.group(1)
        part2_inch_str = m.group(2)
        if part1_mm in mm_to_inch:
            eqs.add(f"dn{part1_mm}*{part2_inch_str}")
            eqs.add(f"{part1_mm}*{part2_inch_str}")
        return eqs

    # Case 1: 'dn' value, e.g., 'dn25'，扩展为dn25、3/4"、25
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
    # --- BUGFIX: Use the normalized keyword for splitting to handle special chars ---
    norm_kw = normalize_material(keyword)
    # 材质
    material_tokens = re.findall(r'pvc-u|ppr|pe|pp|hdpe|pvc conduit|pert', norm_kw)
    # 数字 (修正：匹配包括小数在内的完整数字)
    digit_tokens = re.findall(r'\d+(?:\.\d+)?', norm_kw)
    # 中文同义词整体切分
    chinese_tokens = split_with_synonyms(norm_kw) # Was using keyword, now uses norm_kw
    return material_tokens, digit_tokens, chinese_tokens

def expand_keyword_with_synonyms(keyword: str) -> list[str]: #分词前用
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
    expanded_keywords = expand_keyword_with_synonyms(keyword.strip()) #对关键词做同义词扩展
    all_results = {} # Use dict to store unique results with the best score
    
    # 新增：定义异径相关词
    special_words = {"异径","变径","内丝","内螺纹"}
    for kw in expanded_keywords:
        material_tokens, _, chinese_tokens = classify_tokens(kw) #材质相关的token和其他所有分出来的token

        #包含数字的 token（规格相关，如 "dn20"、"20"、"1/2"）
        query_size_tokens = {t for t in chinese_tokens if re.search(r'\d', t) and not t.endswith('°')}
        #不包含数字的 token（名称、材质相关）
        query_text_tokens = {t for t in chinese_tokens if not (re.search(r'\d', t) and not t.endswith('°'))}

        # 判断用户输入是否包含异径相关词
        is_query_special_words = any(word in kw for word in special_words)

        # 1. 为每个查询规格，创建一个包含所有等价写法的集合
        query_spec_equivalents = {}
        query_material = material_tokens[0] if material_tokens else None
        for token in query_size_tokens:
            query_spec_equivalents[token] = expand_token_with_synonyms_and_units(token, material=query_material)

        for row in df.itertuples(index=False):
            row_identifier = getattr(row, "Describrition", str(row)) 
            raw_text = str(getattr(row, field, ""))
            normalized_text = normalize_material(raw_text).lower()

            # 新增：判断产品描述是否包含异径相关词
            is_row_yijing = any(word in normalized_text for word in special_words)

            # --- 新增等径优先逻辑（无论严格或模糊） ---
            if not is_query_special_words and is_row_yijing:
                continue  # 用户没查异径，但产品是异径，跳过

            if not all(m.lower() in normalized_text for m in material_tokens):
                continue

            # --- REWRITTEN SEARCH LOGIC ---

            product_all_tokens = split_with_synonyms(raw_text)
            product_material_tokens = re.findall(r'pvc|ppr|pe|pp|hdpe|pb|pert', normalized_text)
            product_material = product_material_tokens[0] if product_material_tokens else None
            product_specs_tokens = {t for t in product_all_tokens if re.search(r'\d', t)}
            # 规格标准化集合
            canonical_product_specs = {
                next((eq for eq in expand_token_with_synonyms_and_units(t, product_material) if eq.startswith('dn')), t)
                for t in product_specs_tokens
            }
            canonical_query_specs = {
                next((eq for eq in expand_token_with_synonyms_and_units(t, query_material) if eq.startswith('dn')), t)
                for t in query_size_tokens
            }

            # 1. Count size matches
            size_hits = 0
            if query_size_tokens:
                for q_spec in query_size_tokens:
                    q_equivalents = expand_token_with_synonyms_and_units(q_spec, material=query_material)
                    if any(eq in product_specs_tokens for eq in q_equivalents):
                        size_hits += 1
            
            # In fuzzy mode, if there are size tokens in query, at least one must match
            if not strict and query_size_tokens and size_hits == 0:
                continue

            # 2. Count text matches
            text_hits = 0
            product_text_lower = normalized_text
            for token in query_text_tokens:
                if token.lower() in product_text_lower:
                    text_hits += 1

            # 3. Apply user's rule: for fuzzy search, at least one text token must match
            if not strict and query_text_tokens and text_hits == 0:
                continue

            # 4. Strict mode check
            if strict:
                # All query tokens must be matched
                if size_hits != len(query_size_tokens) or text_hits != len(query_text_tokens):
                    continue

            # 5. Calculate score
            hit_count = size_hits + text_hits
            total_tokens = len(query_size_tokens) + len(query_text_tokens)

            # 新增：承插/对接类优先1.0MPa/PN10，hit_count加1
            chicai_words = {"承插", "承插式", "对接"}
            is_query_chicai = any(word in kw for word in chicai_words)
            is_row_1mpa = ("1.0mpa" in normalized_text or "pn10" in normalized_text)
            if is_query_chicai and is_row_1mpa:
                hit_count += 1

            # 新增：电熔类优先1.6MPa/PN16，hit_count加1
            dianrong_words = {"电熔"}
            is_query_dianrong = any(word in kw for word in dianrong_words)
            is_row_16mpa = ("1.6mpa" in normalized_text or "pn16" in normalized_text)
            if is_query_dianrong and is_row_16mpa:
                hit_count += 1

            # 新增：HDPE直管优先1.0MPa/PN10且6M，hit_count加1
            hdpe_words = {"pe"}
            zhiguan_words = {"管"}
            is_query_hdpe = any(word in kw.lower() for word in hdpe_words)
            is_query_zhiguan = any(word in kw for word in zhiguan_words)
            is_row_1mpa = ("1.0mpa" in normalized_text or "pn10" in normalized_text)
            is_row_6m = ("6m" in normalized_text)
            if is_query_hdpe and is_query_zhiguan and is_row_1mpa and is_row_6m:
                hit_count += 1

            # 新增：如果规格标准化集合完全一致，hit_count加1
            if canonical_query_specs == canonical_product_specs and canonical_query_specs:
                hit_count += 1

            score = hit_count / total_tokens if total_tokens > 0 else 0

            if score > 0:
                 if row_identifier not in all_results or score > all_results[row_identifier][1]:
                    all_results[row_identifier] = (row, score)

    # Convert the results dict back to a list
    final_results = list(all_results.values())
    final_results.sort(key=lambda x: x[1], reverse=True)

    if return_score:
        return final_results
    else:
        return [res[0] for res in final_results]

def prioritize_liansu(df):
    # 新增一列，联塑为1，其它为0
    df['_liansu_priority'] = df['Describrition'].str.contains('联塑').astype(int)
    # 联塑优先，其它次之，原顺序不变
    df = df.sort_values('_liansu_priority', ascending=False).drop('_liansu_priority', axis=1)
    return df

def format_row(i, out_df):
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
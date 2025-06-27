import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text

# — 数据库连接配置 —
db_user     = 'root'
db_password = 'xzy20010506'
db_host     = 'localhost'
db_port     = '3306'
db_name     = 'my_database'
engine      = create_engine(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
)

# — 同义词映射，只针对描述搜索生效 —
SYNONYMS = {
    "大小头": "异径直通",
    "异径套": "异径直通",
    "直接头": "直通",
    "直接": "直通"
}

# — 初始化 Session State —
for key, default in [
    ("cart",      []),
    ("last_out",  pd.DataFrame()),
    ("to_cart",   []),
    ("to_remove", [])
]:
    if key not in st.session_state:
        st.session_state[key] = default

def add_to_cart_callback():
    for i in st.session_state.to_cart:
        st.session_state.cart.append(
            st.session_state.last_out.loc[i].to_dict()
        )
    st.session_state.to_cart = []

def remove_from_cart():
    idxs = set(st.session_state.to_remove)
    st.session_state.cart = [
        item for i,item in enumerate(st.session_state.cart)
        if i not in idxs
    ]
    st.session_state.to_remove = []

# — mm ↔ inch 映射表 —
mm_to_inch = {
    "20":  '1/2"',
    "25":  '3/4"',
    "32":  '1"',
    "50":  '1-1/2"',
    "63":  '2"',
    "75":  '2-1/2"',
    "90":  '3"',
    "110": '4"',
}
inch_to_mm = {v:k for k,v in mm_to_inch.items()}

# — 工具函数 —
def normalize_text(s: str) -> str:
    return re.sub(r'[^0-9A-Za-z\u4e00-\u9fa5]', '', str(s).lower())

def extract_numbers(s: str) -> list[str]:
    text = str(s).replace('＂','"')
    raw  = re.findall(r'\d+-\d+\/\d+|\d+\/\d+|\d+\.\d+|\d+', text)
    out: list[str] = []
    for r in raw:
        if '-' in r and '/' in r:
            w, frac = r.split('-',1)
            n, d    = frac.split('/',1)
            try:
                val = int(w) + int(n)/int(d)
                out.append(str(round(val,2)))  # e.g. "1.5"
            except:
                pass
        else:
            out.append(r)
    return out

def expand_spec_numbers(numbers: list[str]) -> set[str]:
    s = set(numbers)
    dec2mm: dict[str,str] = {}
    for mm, i_str in mm_to_inch.items():
        for d in extract_numbers(i_str):
            dec2mm[d] = mm
    for n in numbers:
        c = n.replace('"','')
        if c in mm_to_inch:
            s.add(mm_to_inch[c].replace('"',''))
        if c + '"' in inch_to_mm:
            s.add(inch_to_mm[c + '"'])
        if c in inch_to_mm:
            s.add(inch_to_mm[c])
        if c in dec2mm:
            s.add(dec2mm[c])
    return s

st.title("🔍 产品查询系统")

keyword    = st.text_input("请输入关键词（产品名称或描述片段）")
mat_kw     = st.text_input("物料号搜索（只匹配 Material 列）")
qty        = st.number_input("请输入数量", min_value=1, value=1)
price_type = st.selectbox("请选择使用的价格字段", ["出厂价_含税","出厂价_不含税"])

if st.button("查询"):
    df      = pd.read_sql(text("SELECT * FROM product"), engine)
    results = []

    # —— 物料号查询分支 —— 
    if mat_kw.strip():
        pat = mat_kw.strip().lower()
        mask = df["Material"].astype(str).str.lower().str.contains(pat)
        for row in df[mask].itertuples(index=False):
            price = getattr(row, price_type, 0) or 0
            results.append({
                "物料编号": getattr(row,"Material",""),
                "产品描述": getattr(row,"Description",""),
                "单价":     price,
                "数量":     float(qty),
                "总价":     price * float(qty)
            })

    # —— 描述 + 数字 分支 —— 
    else:
        raw0 = keyword.strip()
        raw  = raw0.lower()

        # 1) 同义词替换（只针对描述匹配用的 raw）
        used_alias = []
        for alias, std in SYNONYMS.items():
            if alias in raw0:
                used_alias.append(alias)
                raw = raw.replace(alias, std)

        # 2) 提取用户输入的数字，并扩展 mm↔inch
        user_nums = extract_numbers(raw)                   # e.g. ["50"] or ["1.5"]
        user_set  = set(user_nums) | expand_spec_numbers(user_nums)

        # 是否带分数/小数
        need_both = any(('/' in n or '-' in n) for n in user_nums)

        # 3) 构造文字关键词（不含数字）
        toks   = re.findall(r'[A-Za-z]+|[\u4e00-\u9fa5]+', raw)
        txt_kw = []
        for t in toks:
            if re.search(r'[\u4e00-\u9fa5]', t):
                txt_kw += list(t)
            else:
                txt_kw.append(t)
        # 角度 / 压力
        txt_kw += re.findall(r'(\d+)(?=°)', raw)
        txt_kw += re.findall(r'\d+(?:\.\d+)?mpa', raw)

        # 4) 遍历产品库
        for row in df.itertuples(index=False):
            desc = str(getattr(row,"Description",""))
            norm = normalize_text(desc)

            # 数字过滤：
            if user_set:
                row_nums = extract_numbers(desc)
                row_set  = expand_spec_numbers(row_nums)
                if need_both:
                    # 带分数的，mm 和 inch 都要
                    if not user_set.issubset(row_set):
                        continue
                else:
                    # 纯整数，只要有一个就算
                    if not (user_set & row_set):
                        continue

            # 文字过滤（同义词也要替换到 norm 里）
            for alias, std in SYNONYMS.items():
                norm = norm.replace(
                    normalize_text(alias),
                    normalize_text(std)
                )
            ext = "".join(expand_spec_numbers(extract_numbers(desc)))
            combined = norm + ext
            if not all(k in combined for k in txt_kw):
                continue

            price = getattr(row, price_type, 0) or 0
            results.append({
                "物料编号": getattr(row,"Material",""),
                "产品描述": desc,
                "单价":     price,
                "数量":     float(qty),
                "总价":     price * float(qty)
            })

    st.session_state.last_out = pd.DataFrame(results)
    if not results:
        st.info("⚠️ 未查询到相关产品，请尝试更换关键词或检查单位格式。")

# —— 展示 & 加入购物车 —— 
if not st.session_state.last_out.empty:
    st.markdown("### 查询结果")
    st.dataframe(st.session_state.last_out)
    st.multiselect(
        "✅ 请选择要加入产品框的行",
        options=list(st.session_state.last_out.index),
        format_func=lambda i: st.session_state.last_out.loc[i,"产品描述"],
        key="to_cart"
    )
    st.button("添加到产品框", on_click=add_to_cart_callback, key="add_cart_btn")

# —— 购物车 & 删除 —— 
if st.session_state.cart:
    st.markdown("## 🛒 产品框")
    cdf = pd.DataFrame(st.session_state.cart)
    st.dataframe(cdf)

    st.markdown("### ❌ 删除产品框中的条目")
    st.multiselect(
        "✅ 请选择要删除的行",
        options=list(cdf.index),
        format_func=lambda i: cdf.loc[i,"产品描述"],
        key="to_remove"
    )
    st.button("删除所选", on_click=remove_from_cart, key="del_cart_btn")

    total = cdf["总价"].sum()
    st.success(f"产品框总价合计：{total:,.2f}")

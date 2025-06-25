import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text

# — 数据库连接配置 —
db_user = 'root'
db_password = 'xzy20010506'
db_host = 'localhost'
db_port = '3306'
db_name = 'my_database'
engine = create_engine(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
)

# — 初始化购物车 和 上次查询结果 & 选择缓存 —
if "cart" not in st.session_state:
    st.session_state.cart = []
if "last_out" not in st.session_state:
    st.session_state.last_out = pd.DataFrame()
if "to_cart" not in st.session_state:
    st.session_state.to_cart = []

def add_to_cart_callback():
        # 把当前多选框里选中的行加入购物车
        for idx in st.session_state.to_cart:
            st.session_state.cart.append(
            st.session_state.last_out.loc[idx].to_dict()
        )
    # 清空多选，让下次 multiselect 自动重置为空
        st.session_state.to_cart = []

mm_to_inch = {
    "20": '1/2"',   "25": '3/4"',   "32": '1"',
    "50": '1-1/2"', "63": '2"',     "75": '2-1/2"',
    "90": '3"',     "110": '4"'
}
inch_to_mm = {v: k for k, v in mm_to_inch.items()}

# — 同义词映射：
SYNONYMS = {
    "冷水管": "冷给水直管",
    "冷给水管": "冷给水直管",
    # 如果后续还有别的别名，可以继续加
}
# — 文本规范化 —
def normalize_text(s):
    s = str(s).replace('＂', '"')
    return re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', '', s.lower())


# — 提取数字（支持 2-1/2、1/2、全角＂）—
def extract_numbers(s):
    text = str(s).replace('＂', '"')
    raw = re.findall(r'\d+-\d+\/\d+|\d+\/\d+|\d+\.\d+|\d+', text)
    out = []
    for r in raw:
        if '-' in r and '/' in r:
            whole, frac = r.split('-', 1)
            num, den = frac.split('/')
            try:
                val = int(whole) + int(num)/int(den)
                out.append(str(round(val, 2)))
            except:
                pass
        else:
            out.append(r)
    return out

# — 扩展数字（mm↔inch↔小数inch）—
def expand_spec_numbers(numbers):
    s = set(numbers)
    dec2mm = {}
    for mm, inch_str in mm_to_inch.items():
        for d in extract_numbers(inch_str):
            dec2mm[d] = mm
    for n in numbers:
        c = n.replace('"','').replace('＂','')
        if c in mm_to_inch:
            s.add(mm_to_inch[c].replace('"',''))
        if c+'"' in inch_to_mm:
            s.add(inch_to_mm[c+'"'])
        if c in inch_to_mm:
            s.add(inch_to_mm[c])
        if c in dec2mm:
            s.add(dec2mm[c])
    return s

# — 全词匹配函数 —
def all_keywords_in_text(keywords, text):
    return all(k in text for k in keywords)

st.title("🔍 产品关键词智能查询系统")

# — 用户输入 —
keyword    = st.text_input("请输入关键词（产品名称或描述片段）")
qty        = st.number_input("请输入数量", min_value=1, value=1)
price_type = st.selectbox("请选择使用的价格字段", ["出厂价_含税","出厂价_不含税"])

# — 查询按钮：只更新 last_out 和 清空 to_cart —
if st.button("查询"):
    df = pd.read_sql(text("SELECT * FROM product"), engine)

    raw = keyword.strip().lower()
    toks    = re.findall(r'[A-Za-z]+|[\u4e00-\u9fa5]+', raw)
    base_kw = []
    for tok in toks:
        if re.search(r'[\u4e00-\u9fa5]', tok):
            # 中文串拆成单字
            base_kw += list(tok)
        else:
            # 英文整词
            base_kw.append(tok)

    # —— 数字 & 单位扩展 —— 
    nums   = extract_numbers(raw)
    ext_mm = [n for n in expand_spec_numbers(nums) if n.isdigit() and n not in nums]

    # —— 角度提取 —— 
    angle = re.findall(r'(\d+)(?=°)', raw)

    # 最终关键词
    keywords = list(set(base_kw + ext_mm + angle))
    st.write("🔑 最终 keywords:", keywords)
    
    # 5) 在库里做全词匹配
    results = []
    for row in df.itertuples(index=False):
        desc     = str(getattr(row, "Description",""))
        norm     = normalize_text(desc)
        dnums    = extract_numbers(desc)
        dext     = expand_spec_numbers(dnums)
        combined = norm + "".join(dext)

        if all_keywords_in_text(keywords, combined):
            price = getattr(row, price_type, 0) or 0
            q     = float(qty)
            results.append({
                "产品描述": desc,
                "单价": price,
                "数量": q,
                "总价": price * q
            })

    # 保存查询结果 & 重置多选
    st.session_state.last_out = pd.DataFrame(results)

    if st.session_state.last_out.empty:
        st.info("⚠️ 未查询到相关产品，请尝试更换关键词或检查单位格式。")
    
# — 如果有查询结果，就显示 & 选行加入购物车 —
if not st.session_state.last_out.empty:
    st.markdown("### 查询结果")
    st.dataframe(st.session_state.last_out)
    
    # 多选框：绑定到 session_state.to_cart
    st.multiselect(
        "✅ 请选择要加入产品框的行",
        options=list(st.session_state.last_out.index),
        format_func=lambda i: st.session_state.last_out.loc[i, "产品描述"],
        key="to_cart"
    )
    
    st.button(
        "添加到产品框",
        on_click=add_to_cart_callback,
        key="add_cart_btn"
    )

# — 最终展示当前购物车及总价 —
if st.session_state.cart:
    st.markdown("## 🛒 产品框")
    cart_df = pd.DataFrame(st.session_state.cart)
    st.dataframe(cart_df)
    st.success(f"产品框总价合计：{cart_df['总价'].sum():,.2f}")
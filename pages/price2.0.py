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
    # 全部小写，去除空格、连字符、全角转半角
    s = s.lower()
    s = s.replace('－', '-').replace('—', '-').replace('–', '-')
    s = re.sub(r'[\s_]', '', s)  # 去空格和下划线
    s = s.replace('（', '(').replace('）', ')')
    # 全角转半角
    s = ''.join([chr(ord(c)-65248) if 65281 <= ord(c) <= 65374 else c for c in s])
    # 常见同义写法归一化
    s = s.replace('pp r', 'ppr').replace('pp-r', 'ppr').replace('p-pr', 'ppr').replace('p.p.r', 'ppr')
    s = s.replace('p-vc', 'pvc').replace('p.v.c', 'pvc').replace('p v c', 'pvc').replace('p.v.c.', 'pvc')
    return s

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
SYNONYMS = {
    "大小头":"异径直通",
    "异径套":"异径直通",
    "直接头":"直通",
    "直接":"直通"
    }
mm_to_inch = {"20": '1/2"', "25": '3/4"',
              "32": '1"', "50": '1-1/2"',
              "63": '2"', "75": '2-1/2"',
              "90": '3"', "110": '4"'}
inch_to_mm = {v:k for k,v in mm_to_inch.items()}

def normalize_text(s: str) -> str:
    return re.sub(r'[^0-9A-Za-z一-龥]', '', str(s).lower())

def extract_numbers(s: str) -> list[str]:
    raw = re.findall(r'\d+-\d+\/\d+|\d+\/\d+|\d+\.\d+|\d+', str(s).replace('＂','"'))
    out = []
    for r in raw:
        if '-' in r and '/' in r:
            w, frac = r.split('-',1)
            n, d    = frac.split('/',1)
            try: out.append(str(round(int(w) + int(n)/int(d),2)))
            except: pass
        else:
            out.append(r)
    return out

def expand_spec_numbers(nums: list[str]) -> set[str]:
    s = set(nums); dec2mm = {}
    for mm,i_str in mm_to_inch.items():
        for d in extract_numbers(i_str):
            dec2mm[d] = mm
    for n in nums:
        c = n.replace('"','')
        if c in mm_to_inch:       s.add(mm_to_inch[c].replace('"',''))
        if c+'"' in inch_to_mm:   s.add(inch_to_mm[c+'"'])
        if c in inch_to_mm:       s.add(inch_to_mm[c])
        if c in dec2mm:           s.add(dec2mm[c])
    return s

def apply_synonym_and_unit_maps(s: str) -> set:
    """返回所有同义词和单位映射后的字符串集合"""
    s = s.lower()
    results = {s}
    # 同义词映射
    for a, b in SYNONYMS.items():
        if a in s:
            results.add(s.replace(a, b))
        if b in s:
            results.add(s.replace(b, a))
    # 单位映射
    for mm, inch in mm_to_inch.items():
        if mm in s:
            results.add(s.replace(mm, inch))
        if inch in s:
            results.add(s.replace(inch, mm))
    return results

def all_mapped_chars(s: str) -> str:
    # 应用同义词和单位映射，返回所有映射后的字符串拼接
    s = s.lower()
    mapped = [s]
    # 同义词映射
    for a, b in SYNONYMS.items():
        if a in s:
            mapped.append(s.replace(a, b))
        if b in s:
            mapped.append(s.replace(b, a))
    # 单位映射
    for mm, inch in mm_to_inch.items():
        if mm in s:
            mapped.append(s.replace(mm, inch))
        if inch in s:
            mapped.append(s.replace(inch, mm))
    return "".join(mapped)

def extract_size_tokens(size: str):
    # 提取所有数字、分数、带引号的尺寸
    # 例：'dn20 1/2"' -> ['20', '1/2"']
    tokens = []
    # 匹配如 1/2" 3/4" 20 25 110 等
    tokens += re.findall(r'\d+/\d+"|\d+\.?\d*', size)
    return tokens

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

# 页面切换逻辑
if page == "查询产品":
    st.header("产品查询系统")
    df = load_data()

    
    # 三列布局
    c1, c3 = st.columns([6,1])
    with c1:
        keyword = st.text_input("关键词（名称、规格、材质可一起输入）")
    with c3:
        qty = st.number_input("数量", min_value=1, value=1)
    mat_kw = st.text_input("物料号搜索")
    price_type = st.selectbox("价格字段", ["出厂价_含税","出厂价_不含税"])

    if st.button("查询"):
        results = []
        if mat_kw.strip():
            pat = mat_kw.lower().strip()
            mask = df["Material"].astype(str).str.lower().str.contains(pat)
            for row in df[mask].itertuples(index=False):
                price = getattr(row, price_type, 0) or 0
                total = (float(str(price).replace(',', '')) * qty
                         if isinstance(price, str)
                         else float(price) * qty)
                results.append({
                    "物料编号": getattr(row, "Material", ""),
                     "产品描述": getattr(row, "Describrition", ""),
                     "单价": price,
                     "数量": qty,
                     "总价": total
                 })
        else:
            raw0 = keyword.strip().lower()
            for a,s in SYNONYMS.items():
                raw0 = raw0.replace(a,s)
            user_nums = extract_numbers(raw0)
            user_set  = set(user_nums) | expand_spec_numbers(user_nums)
            need_both = any('/' in n or '-' in n for n in user_nums)

            # 检查是否需要强制匹配pvc/ppr
            norm_keyword = normalize_material(keyword)
            require_pvc = "pvc" in norm_keyword
            require_ppr = "ppr" in norm_keyword  # 现在会匹配所有PP-R的变体
            #  名称拆单字
            tokens = list(keyword.strip()) if keyword.strip() else []


            for row in df.itertuples(index=False):
                desc = str(getattr(row, "Describrition", ""))
                norm_desc = normalize_text(desc)
                # PVC / PPR 必须命中
                if require_pvc and "pvc" not in norm_desc:
                    continue
                if require_ppr and "ppr" not in norm_desc:
                    continue

                # 分词 + 规格必须都包含
                if tokens and not all(t in norm_desc for t in tokens):
                    continue

                row_set = expand_spec_numbers(extract_numbers(desc))
                if user_set:
                    if need_both and not user_set.issubset(row_set):
                        continue
                    if not need_both and not (user_set & row_set):
                        continue

                price = getattr(row, price_type, 0) or 0
                total = float(str(price).replace(',','')) * qty
                results.append({
                    "物料编号": getattr(row, "Material", ""),
                     "产品描述": desc,
                     "单价": price,
                     "数量": qty,
                     "总价": total
                 })
        st.session_state.last_out = pd.DataFrame(results)

    # 查询结果展示（始终渲染）
    out_df = st.session_state.get("last_out", pd.DataFrame())
    st.subheader("📊 查询结果")
    if not out_df.empty:
        st.dataframe(out_df, use_container_width=True)
        to_cart = st.multiselect(
            "选择要加入购物车的行",
            options=list(out_df.index),
            format_func=lambda i: out_df.loc[i, "产品描述"],
            key="to_cart"
        )
        if st.button("添加到购物车", key="add_cart"):
            add_to_cart()
            st.success("✅ 已加入购物车")
            st.rerun()
    else:
        st.warning("⚠️ 未查询到符合条件的产品")

    # 购物车显示（只在查询界面显示）
    if st.session_state.cart:
        cart_df = pd.DataFrame(st.session_state.cart)
        st.dataframe(cart_df, use_container_width=True)
        to_remove = st.multiselect(
            "选择要删除的购物车条目",
            options=list(cart_df.index),
            format_func=lambda i: cart_df.loc[i,"产品描述"],
            key="to_remove"
        )
        if st.button("删除所选", key="del_cart_bottom"):
            remove_from_cart()
            st.rerun()
            st.success(f"购物车总价：{cart_df['总价'].sum():,.2f}")
    else:
        st.info("购物车为空")

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
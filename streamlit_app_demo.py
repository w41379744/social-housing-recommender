# 社宅推薦系統 Demo - Streamlit 版本

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb

# --- 載入模型與資料 ---
@st.cache_resource
def load_model():
    model = lgb.Booster(model_file='model.txt')  # 載入訓練好的模型
    return model

@st.cache_data
def load_data():
    df = pd.read_excel("house_items.xlsx")  # Excel 格式
    return df

model = load_model()
df_items = load_data()

# --- 網頁 UI ---
st.title("🏠 社宅推薦系統 Demo")
st.markdown("請輸入以下條件，系統將推薦適合的社宅房源")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("承租人年齡", min_value=18, max_value=99, value=25)
    is_vul = st.radio("是否為弱勢身分", ["是", "否"]) == "是"

with col2:
    county = st.selectbox("縣市", df_items["縣市"].unique())
    district = st.selectbox("鄉鎮市區", df_items[df_items["縣市"] == county]["鄉鎮市區"].unique())

rent_max = st.slider("租金預算上限", 0, 70000, 20000, step=1000)

# --- 推薦邏輯 ---
def recommend(age, is_vul, county, district, model, df_items, topk=5):
    df = df_items.copy()
    df = df[(df["縣市"] == county) & (df["鄉鎮市區"] == district)]
    if df.empty:
        return pd.DataFrame()

    # 必要欄位補齊
    df["承租人年齡"] = age
    df["承租人是否為弱勢"] = int(is_vul)
    df["承租人性別"] = -1
    df["出租人年齡"] = -1  # 預設值

    # 特徵工程（與模型訓練一致）
    df["簽約租金x屋齡"] = df["簽約租金"] * df["屋齡"]
    df["簽約租金x坪數"] = df["簽約租金"] * df["實際使用坪數"]
    df["屋齡x幾房"] = df["屋齡"] * df["幾房"]
    df["坪數x幾房"] = df["實際使用坪數"] * df["幾房"]

    df = df.fillna(-1)

    # 模型特徵
    features = [
        "簽約租金", "屋齡", "幾房", "幾廳", "幾衛浴", "實際使用坪數",
        "出租人年齡", "承租人性別", "承租人是否為弱勢", "承租人年齡",
        "簽約租金x屋齡", "簽約租金x坪數", "屋齡x幾房", "坪數x幾房"
    ]

    df["score"] = model.predict(df[features])
    df = df[df["簽約租金"] <= rent_max]
    return df.sort_values("score", ascending=False).head(topk)

# --- 執行推薦 ---
if st.button("推薦房源"):
    recs = recommend(age, is_vul, county, district, model, df_items)
    if recs.empty:
        st.warning("查無推薦結果，請嘗試放寬條件")
    else:
        st.subheader("推薦結果：")
        for _, row in recs.iterrows():
            st.markdown(f"#### 💡 {row['建物型態']}｜{row['幾房']}房｜{row['實際使用坪數']}坪")
            st.markdown(f"📍 {row['縣市']} {row['鄉鎮市區']}")
            st.markdown(f"💰 租金：NTD {int(row['簽約租金'])}/月")
            st.markdown("---")

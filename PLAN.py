import os
import json
import re
import base64
import datetime
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
import requests
from io import StringIO
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

# ---------- 1. 初始化與環境設定 ----------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# GitHub 倉儲設定
REPO_OWNER = "ss900371tw"
REPO_NAME = "NINEAI"
FILE_PATH = "RAG.csv"

if not GOOGLE_API_KEY:
    st.error("請在 .env 檔案中設定 GOOGLE_API_KEY")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# 安全性設定
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# 初始化模型
model = genai.GenerativeModel(
    model_name="gemini-2.5-pro", # 修正為官方正確版本號
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.1,
    },
    safety_settings=SAFETY_SETTINGS
)

# ---------- 2. 原則定義 ----------
# ---------- 2. 原則定義 ----------
TRANSPARENCY_9 = [
    {"title": "介入詳情及輸出", "desc": "需清楚定義產出的具體內容，如標記位置、風險評分（0-100 分）或分類建議，指引醫師解讀結果。"},
    {"title": "介入目的", "desc": "說明 AI 的臨床用途（如輔助診斷、分流或篩查）及其預期解決的具體臨床痛點。"},
    {"title": "警告與範圍外使用", "desc": "明確限制條件，告知醫師不適用情境（如特定機型、非適應症族群），並強調不得獨立作為診斷工具。"},
    {"title": "開發詳情及輸入特徵", "desc": "揭露訓練資料來源、特徵維度（如年齡、像素、密度、腫塊）及採用的算法架構（如 CNN）。"},
    {"title": "確保公平性的過程", "desc": "詳述如何檢查並減少演算法偏見，確保在不同種族、性別或年齡層表現的一致性。"},
    {"title": "外部驗證過程", "desc": "展示在獨立真實世界數據上的表現，包含跨中心數量、硬體製造商分佈及組織學類型。"},
    {"title": "量化表現指標", "desc": "提供靈敏度、特異性、AUC 等具體統計數據，作為醫師評估系統效能的基準。"},
    {"title": "持續維護與監控", "desc": "描述部署後的技術支援、監控團隊及更新計畫，確保系統在臨床現場的穩定性。"},
    {"title": "更新與持續驗證計畫", "desc": "規定再訓練頻率與定期驗證門檻，以應對醫療環境變遷導致的性能波動。"}
]

GOVERNANCE_2 = [
    {"title": "可解釋性分析", "desc": "可解釋性分析在醫療人工智慧中是指用來解釋和理解人工智慧模型如何做出預測或決策的技術和方法。這在醫療領域中至關重要，因為透明性和信任對於人工智慧工具的採用是必不可少的。其目標是提供對人工智慧系統決策過程的洞見，確保臨床醫師能夠理解和驗證其輸出結果。"},
    {"title": "AI生命週期管理", "desc": "AI 生命週期循環監測有效性在臨床醫學的應用涉及到對人工智慧（AI）系統在整個生命週期中的有效性進行持續的監測和評估。這一過程不僅包括 AI 系統的開發和部署階段，還涵蓋了後續的運行、維護和改進。這樣的監測確保了 AI系統在實際臨床環境中的表現能夠持續符合預期，並且能夠適應隨時間變化的醫療需求和資料特性，實施定期的性能監控計畫"}
]

# ---------- 3. 功能函式 ----------

def get_rag_df_from_github():
    """從 GitHub 讀取目前的 RAG 庫"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        content = base64.b64decode(res.json()['content']).decode('utf-8')
        
        # --- 核心修正處 ---
        if not content.strip():  # 如果檔案內容是空的
            return pd.DataFrame(columns=["Timestamp", "Principle", "UserFeedback", "OriginalSummary"])
        
        try:
            return pd.read_csv(StringIO(content))
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=["Timestamp", "Principle", "UserFeedback", "OriginalSummary"])
    
    return pd.DataFrame(columns=["Timestamp", "Principle", "UserFeedback", "OriginalSummary"])



def generalize_feedback(specific_feedback):
    prompt = f"""
    使用者針對醫療 AI 審查提供了具體修正建議：'{specific_feedback}'
    請以純文字將其轉為一條通用的審查原則，使其能適用於其他不同的計畫書。
    只回傳轉化後的文字，不要有其他解釋。
    """
    response = model.generate_content(prompt)
    return response.text.strip()


    




def update_rag_to_github(timestamp, principle, feedback, original_summary):
    """將回饋存入 GitHub"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

    # 1. 取得現有資料
    df = get_rag_df_from_github()
    res = requests.get(url, headers=headers)
    sha = res.json().get('sha') if res.status_code == 200 else None

    # 2. 加入新列
    new_data = pd.DataFrame([{
        "Timestamp": timestamp,
        "Principle": principle,
        "UserFeedback": feedback,
    }])
    df = pd.concat([df, new_data], ignore_index=True)

    # 3. 轉回 CSV 並推送到 GitHub (使用 pandas 確保格式正確)
    csv_content = df.to_csv(index=False, encoding='utf-8')
    encoded_content = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
    
    payload = {
        "message": f"Update RAG feedback for {principle}",
        "content": encoded_content,
        "sha": sha
    }
    
    put_res = requests.put(url, headers=headers, json=payload)
    return put_res.status_code in [200, 201]

def analyze_item(item, context_text, rag_history=""):
    """執行單項 AI 檢核"""
    prompt = f"""
    你是一位醫療 AI 合規性審查專家。請針對以下原則分析文件內容。
    
    【檢核原則】
    原則：{item['title']}
    定義：{item['desc']}
    
    【歷史修正參考 (RAG)】
    以下是過去人工審查對「{item['title']}」類似內容的修正建議，請將這些經驗納入本次判斷考量：
    {rag_history if rag_history else "尚無歷史參考資料。"}
    
    【文件內容】
    {context_text[:12000]}
    
    請根據上述資訊，以 JSON 格式回覆：
    {{
      "status": "存在" 或 "不存在",
      "summary": "具體做法摘要",
      "suggestion": "缺失建議"
    }}
    """
    try:
        response = model.generate_content(prompt)
        clean_text = re.sub(r"```json\n?|\n?```", "", response.text).strip()
        return json.loads(clean_text)
    except Exception as e:
        return {"status": "檢核錯誤", "summary": f"API 錯誤: {str(e)}", "suggestion": ""}



import numpy as np

def get_embedding(text):
    """將文字轉換為向量 - 修正模型路徑"""
    try:
        # 嘗試使用最通用的 embedding 模型名稱
        result = genai.embed_content(
            model="models/gemini-embedding-001", # 如果 004 不行，001 是穩定首選
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        # 如果還是失敗，拋出錯誤以便調試
        st.error(f"Embedding 錯誤: {e}")
        return [0] * 768  # 回傳零向量避免後續計算崩潰
        
        

def cosine_similarity(v1, v2):
    """計算餘弦相似度"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    
def run_full_analysis(full_text):
    """執行完整分析，包含 9+2 個項目"""
    # 1. 取得歷史 RAG 資料
    rag_df = get_rag_df_from_github()
    
    # 2. 合併所有待檢核項目
    all_items = TRANSPARENCY_9 + GOVERNANCE_2
    
    # 3. 逐項執行分析 (建議使用 ThreadPoolExecutor 加速，或單純迴圈)
    results_t = []
    results_g = []
    
    # 為了方便示範，使用簡單迴圈
    for i, item in enumerate(all_items):
        history = ""
        # 尋找相似歷史經驗
        if not rag_df.empty and "UserFeedback" in rag_df.columns:
            query_text = f"{item['title']}: {item['desc']}"
            try:
                query_vec = get_embedding(query_text)
                rel_rows = rag_df[rag_df["Principle"] == item["title"]].copy()
                
                if not rel_rows.empty:
                    similarities = []
                    for fb in rel_rows["UserFeedback"].tolist():
                        fb_vec = get_embedding(fb)
                        similarities.append(cosine_similarity(query_vec, fb_vec))
                    
                    rel_rows["sim"] = similarities
                    top_3 = rel_rows.sort_values(by="sim", ascending=False).head(3)
                    history = "\n".join([f"- {row['UserFeedback']}" for _, row in top_3.iterrows()])
            except Exception as e:
                st.warning(f"RAG 檢索失敗 ({item['title']}): {e}")

        # 呼叫原本的單項分析函式
        res = analyze_item(item, full_text, rag_history=history)
        
        if i < 9:
            results_t.append(res)
        else:
            results_g.append(res)
            
    return {"t": results_t, "g": results_g}



# ---------- 4. UI 介面 ----------

def main():
    st.set_page_config(page_title="醫療 AI 治理檢核", layout="wide")
    st.title("🛡️ 負責任 AI 自動檢核系統 (RAG 強化版)")

    if 'res_t' not in st.session_state:
        st.session_state['res_t'] = None

    with st.sidebar:
        st.header("1. 檔案讀取")
        pdf_file = st.file_uploader("上傳計畫書 PDF", type="pdf")
        btn = st.button("🚀 開始分析", use_container_width=True)
        st.divider()
        st.info("您的回饋建議將存入 AI 知識庫，用於強化未來分析結果。")

    if pdf_file and btn:
        with st.spinner("正在讀取檔案並檢索歷史經驗..."):
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            full_text = "\n".join([page.get_text() for page in doc])
            results = run_full_analysis(full_text)
            st.session_state['res_t'] = results['t']
            st.session_state['res_g'] = results['g']

    # 顯示結果
    if st.session_state['res_t']:
        st.subheader("📊 九大透明性原則")
        t_data = st.session_state['res_t']
        for r in range(3):
            cols = st.columns(3)
            for c in range(3):
                idx = r * 3 + c
                if idx < len(t_data):
                    item = t_data[idx]
                    with cols[c]:
                        color = "green" if item['status'] == "存在" else "red"
                        st.markdown(f"### {idx+1}. {TRANSPARENCY_9[idx]['title']}")
                        st.markdown(f"**狀態：** :{color}[{item['status']}]")
                        st.info(item['summary'])
                        if item['suggestion']:
                            st.warning(f"💡 建議：{item['suggestion']}")

        st.divider()
        st.subheader("📋 核心治理指標")
        g_data = st.session_state['res_g']
        df_g = pd.DataFrame([{
            "評估項目": GOVERNANCE_2[i]['title'],
            "狀態": d['status'],
            "摘要": d['summary'],
            "建議": d['suggestion']
        } for i, d in enumerate(g_data)])
        st.table(df_g)

        # 回饋收集
        st.divider()
        st.subheader("📝 訓練 AI 的判斷經驗 (RAG)")
        with st.form("rag_feedback_form"):
            all_titles = [i['title'] for i in (TRANSPARENCY_9 + GOVERNANCE_2)]
            col1, col2 = st.columns([1, 2])
            with col1:
                selected_title = st.selectbox("選擇要修正的項目", all_titles)
            with col2:
                user_comment = st.text_area("修正建議 (AI 哪裡看錯了？)", placeholder="例如：『應加強對表格內數據的識別』或『此類資訊通常出現在附件的技術規格中』")
            submit_rag = st.form_submit_button("✅ 送出經驗並優化未來分析")
            generalized_comment = generalize_feedback(user_comment)

            if submit_rag:
                if not GITHUB_TOKEN:
                    st.error("請檢查 GITHUB_TOKEN 設定。")
                elif not user_comment:
                    st.warning("請填寫建議。")
                else:
                    all_results = st.session_state['res_t'] + st.session_state['res_g']
                    idx = all_titles.index(selected_title)
                    orig_sum = all_results[idx]['summary']
                    
                    with st.spinner("同步至 GitHub 中..."):
                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if update_rag_to_github(now, selected_title, generalized_comment, orig_sum):
                            st.success("回饋成功！下次分析將參考此經驗。")
                        else:
                            st.error("寫入失敗，請確認 Token 權限。")

if __name__ == "__main__":
    main()

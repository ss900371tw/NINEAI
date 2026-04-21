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
TRANSPARENCY_9 = [
    {"title": "介入詳情及輸出", "desc": "詳細說明 AI 產品的技術規格，包含模型架構、輸入數據的格式要求以及輸出的具體臨床含義。"},
    {"title": "介入目的", "desc": "明確界定該 AI 在臨床工作流中的角色，包含預期用途與適應症。"},
    {"title": "警告範圍外使用", "desc": "列出該產品的禁忌症與技術極限，即在何種情況下 AI 可能失效或產生錯誤誤導。"},
    {"title": "開發詳情及輸入", "desc": "揭露訓練資料集的特徵，包括數據來源、入選與排除標準、資料分布情形及標註流程。"},
    {"title": "開發公平性過程", "desc": "說明開發團隊如何識別並緩解潛在的演算法偏差，確保模型對不同群體表現一致。"},
    {"title": "外部驗證過程", "desc": "使用未參與開發過程的獨立資料集進行效能測試，以驗證模型的泛化能力。"},
    {"title": "表現量化指標", "desc": "提供多維度的效能評估報告，而非單一的準確率。"},
    {"title": "實施與持續維護", "desc": "描述產品部署後的監控機制，包含系統整合需求、使用者教育訓練及異常回報流程。"},
    {"title": "更新與公平性評估", "desc": "規範模型版本迭代的流程，確保在軟體更新或重新訓練後，模型的效能與安全性不會倒退。"}
]

GOVERNANCE_2 = [
    {"title": "可解釋性分析", "desc": "利用技術讓醫師理解模型決策，確保輸出可驗證。"},
    {"title": "AI生命週期管理", "desc": "從開發到部署的全程風險評估與合規性監測。"}
]

# ---------- 3. 功能函式 ----------

def get_rag_df_from_github():
    """從 GitHub 讀取目前的 RAG 庫"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        content = base64.b64decode(res.json()['content']).decode('utf-8')
        return pd.read_csv(StringIO(content))
    return pd.DataFrame(columns=["Timestamp", "Principle", "UserFeedback", "OriginalSummary"])

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
        "OriginalSummary": original_summary
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

def run_full_analysis(full_text):
    """執行完整分析流程"""
    all_items = TRANSPARENCY_9 + GOVERNANCE_2
    
    # --- 關鍵修改：先載入 RAG 資料 ---
    try:
        rag_df = get_rag_df_from_github()
    except:
        rag_df = pd.DataFrame()

    def process_with_rag(item):
        # 從 CSV 篩選出與目前項目相關的過去建議（取最近 3 筆避免內容過長）
        history = ""
        if not rag_df.empty and "Principle" in rag_df.columns:
            rel_rows = rag_df[rag_df["Principle"] == item["title"]]
            if not rel_rows.empty:
                history = "\n".join([f"- {fb}" for fb in rel_rows["UserFeedback"].tail(3).tolist()])
        
        return analyze_item(item, full_text, rag_history=history)

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_with_rag, all_items))
    
    return {"t": results[:9], "g": results[9:]}

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
        st.info(f"RAG 庫路徑: {REPO_OWNER}/{REPO_NAME}/{FILE_PATH}")

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
                user_comment = st.text_area("修正建議 (AI 哪裡看錯了？)", placeholder="例如：第4頁其實有提到數據來源是北醫...")
            
            submit_rag = st.form_submit_button("✅ 送出經驗並優化未來分析")

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
                        if update_rag_to_github(now, selected_title, user_comment, orig_sum):
                            st.success("回饋成功！下次分析將參考此經驗。")
                        else:
                            st.error("寫入失敗，請確認 Token 權限。")

if __name__ == "__main__":
    main()

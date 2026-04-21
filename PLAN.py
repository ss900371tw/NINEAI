import os
import json
import re
import base64
import datetime
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
import requests
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

# 設定安全性
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# 初始化 Gemini 2.5 Pro
model = genai.GenerativeModel(
    model_name="gemini-2.5-pro",
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.2,
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

def update_rag_to_github(timestamp, principle, feedback, original_summary):
    """透過 GitHub API 更新 RAG.csv"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    # 1. 取得現有檔案內容
    res = requests.get(url, headers=headers)
    sha = None
    content = "Timestamp,Principle,UserFeedback,OriginalSummary\n" # 預設標題
    
    if res.status_code == 200:
        file_data = res.json()
        sha = file_data['sha']
        content = base64.b64decode(file_data['content']).decode('utf-8')
    
    # 2. 格式化新資料列 (處理換行與引號)
    safe_feedback = feedback.replace('"', '""').replace('\n', ' ')
    safe_summary = original_summary.replace('"', '""').replace('\n', ' ')
    new_row = f'"{timestamp}","{principle}","{safe_feedback}","{safe_summary}"\n'
    updated_content = content if content.endswith('\n') else content + '\n'
    updated_content += new_row

    # 3. 推送更新
    encoded_content = base64.b64encode(updated_content.encode('utf-8')).decode('utf-8')
    payload = {
        "message": f"Update RAG feedback for {principle}",
        "content": encoded_content,
        "sha": sha if sha else None
    }
    
    put_res = requests.put(url, headers=headers, json=payload)
    return put_res.status_code in [200, 201]

def analyze_item(item, context_text):
    """單項分析邏輯"""
    prompt = f"""
    你是一位醫療 AI 合規性審查專家。請針對以下原則分析文件內容：
    原則：{item['title']}
    定義：{item['desc']}
    
    文件內容：{context_text[:12000]}
    
    請以 JSON 格式回覆，不需其他解釋：
    {{
      "status": "存在" 或 "不存在",
      "summary": "具體做法摘要（若不存在則寫未見描述）",
      "suggestion": "缺失建議（若存在則留空）"
    }}
    """
    try:
        response = model.generate_content(prompt)
        clean_text = re.sub(r"```json\n?|\n?```", "", response.text).strip()
        return json.loads(clean_text)
    except Exception as e:
        return {"status": "檢核錯誤", "summary": f"API 錯誤: {str(e)}", "suggestion": ""}

def run_full_analysis(full_text):
    all_items = TRANSPARENCY_9 + GOVERNANCE_2
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(lambda x: analyze_item(x, full_text), all_items))
    return {"t": results[:9], "g": results[9:]}

# ---------- 4. UI 介面 (Streamlit) ----------

def main():
    st.set_page_config(page_title="醫療 AI 治理檢核", layout="wide")
    st.title("🛡️ 負責任 AI 自動檢核系統 (Gemini 2.5 Pro)")

    with st.sidebar:
        st.header("1. 檔案讀取")
        pdf_file = st.file_uploader("上傳計畫書 PDF", type="pdf")
        btn = st.button("🚀 開始分析", use_container_width=True)
        st.divider()
        st.info(f"目標 RAG 庫: {REPO_OWNER}/{REPO_NAME}")

    if pdf_file and btn:
        with st.spinner("Gemini 2.5 Pro 正在分析中..."):
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            full_text = "\n".join([page.get_text() for page in doc])
            results = run_full_analysis(full_text)
            st.session_state['res_t'] = results['t']
            st.session_state['res_g'] = results['g']

    # --- 顯示結果 ---
    if 'res_t' in st.session_state:
        st.subheader("📊 九大透明性原則 (九宮格)")
        t_data = st.session_state['res_t']
        
        for r in range(3):
            cols = st.columns(3)
            for c in range(3):
                idx = r * 3 + c
                item = t_data[idx]
                with cols[c]:
                    color = "green" if item['status'] == "存在" else "red"
                    st.markdown(f"### {idx+1}. {TRANSPARENCY_9[idx]['title']}")
                    st.markdown(f"**狀態：** :{color}[{item['status']}]")
                    st.info(item['summary'])
                    if item['suggestion']:
                        st.warning(f"💡 建議：{item['suggestion']}")

        st.divider()
        st.subheader("📋 核心治理指標 (表格)")
        g_data = st.session_state['res_g']
        df_g = pd.DataFrame([{
            "評估項目": GOVERNANCE_2[i]['title'],
            "狀態": d['status'],
            "摘要": d['summary'],
            "建議": d['suggestion']
        } for i, d in enumerate(g_data)])
        st.table(df_g)

        # --- 回饋收集區塊 ---
        st.divider()
        st.subheader("📝 RAG 回饋優化庫")
        st.write("若您發現 AI 分析結果不準確，請在此提供回饋，這將存入 RAG.csv 以優化未來分析。")
        
        with st.form("rag_feedback_form"):
            col_sel, col_text = st.columns([1, 2])
            with col_sel:
                all_titles = [i['title'] for i in (TRANSPARENCY_9 + GOVERNANCE_2)]
                selected_title = st.selectbox("選擇要修正的項目", all_titles)
            with col_text:
                user_comment = st.text_area("您的修正建議：", placeholder="例如：此文件第5頁已有提到外部驗證資料來源...")
            
            submit_rag = st.form_submit_button("✅ 送出回饋至 GitHub")

            if submit_rag:
                if not GITHUB_TOKEN:
                    st.error("找不到 GITHUB_TOKEN，無法儲存回饋。")
                elif not user_comment:
                    st.warning("請填寫回饋內容。")
                else:
                    # 獲取當時 AI 的摘要內容
                    all_results = st.session_state['res_t'] + st.session_state['res_g']
                    idx = all_titles.index(selected_title)
                    orig_sum = all_results[idx]['summary']
                    
                    with st.spinner("正在寫入 GitHub RAG.csv..."):
                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        success = update_rag_to_github(now, selected_title, user_comment, orig_sum)
                        if success:
                            st.success(f"成功！回饋已存入 {FILE_PATH}")
                        else:
                            st.error("寫入失敗，請檢查 GitHub Token 權限。")

if __name__ == "__main__":
    main()

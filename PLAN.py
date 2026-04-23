import os
import json
import re
import base64
import datetime
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
import requests
from io import StringIO, BytesIO # 新增 BytesIO 用於 Excel 處理
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# ---------- 1. 初始化與環境設定 (保持不變) ----------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "ss900371tw"
REPO_NAME = "NINEAI"
FILE_PATH = "RAG.csv"

if not GOOGLE_API_KEY:
    st.error("請在 .env 檔案中設定 GOOGLE_API_KEY")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash", # 建議使用 flash 以提升批次處理速度
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.1,
    },
    safety_settings=SAFETY_SETTINGS
)
# ---------- 2. 原則定義 ----------
TRANSPARENCY_9 = [
    {"title": "介入詳情及輸出", "desc": "需清楚定義模型輸出，如標記位置、風險評分（0-100 分）或分類建議，指引醫師解讀結果。"},
    {"title": "介入目的", "desc": "說明臨床用途（如輔助診斷、分流）及其預期解決痛點。"},
    {"title": "警告與範圍外使用", "desc": "限制不適用情境（如特定機型、非適應症族群），並強調不得獨立作為診斷工具。"},
    {"title": "開發詳情及輸入特徵", "desc": "揭露訓練資料來源、特徵維度（如年齡、性別、影像維度等）及模型架構（如 CNN）。"},
    {"title": "確保公平性的過程", "desc": "詳述如何減少演算法偏見，確保在不同種族、性別或年齡層表現的一致性。"},
    {"title": "外部驗證過程", "desc": "展示單一中心外部驗證或跨中心聯邦驗證在真實數據表現；若為聯邦驗證須詳列中心數量及各院資料量等資訊"},
    {"title": "量化表現指標", "desc": "提供靈敏度、特異性、AUC 等具體統計數據，作為模型效能基準。"},
    {"title": "持續維護與監控", "desc": "描述部署後的技術支援、監控團隊及更新計畫，確保系統在臨床現場穩定性。"},
    {"title": "更新與持續驗證計畫", "desc": "規定再訓練頻率與定期驗證門檻，以應對醫療環境變遷的性能波動。"}
]

GOVERNANCE_2 = [
    {"title": "可解釋性分析", "desc": "可解釋性分析在醫療人工智慧中是指用來解釋和理解人工智慧模型如何做出預測或決策的技術和方法。這在醫療領域中至關重要，因為透明性和信任對於人工智慧工具的採用是必不可少的。其目標是提供對人工智慧系統決策過程的洞見，確保臨床醫師能夠理解和驗證其輸出結果。"},
    {"title": "AI生命週期管理", "desc": "AI 生命週期循環監測有效性在臨床醫學的應用涉及到對人工智慧（AI）系統在整個生命週期中的有效性進行持續的監測和評估。這一過程不僅包括 AI 系統的開發和部署階段，還涵蓋了後續的運行、維護和改進。這樣的監測確保了 AI系統在實際臨床環境中的表現能夠持續符合預期，並且能夠適應隨時間變化的醫療需求和資料特性，實施定期的性能監控計畫"}
]

# ---------- 3. 功能函式 ----------

def get_rag_df_from_github():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        content = base64.b64decode(res.json()['content']).decode('utf-8')
        if not content.strip(): return pd.DataFrame(columns=["Principle", "UserFeedback"])
        try: return pd.read_csv(StringIO(content))
        except: return pd.DataFrame(columns=["Principle", "UserFeedback"])
    return pd.DataFrame(columns=["Principle", "UserFeedback"])

def analyze_item(item, context_text, rag_history=""):
    prompt = f"""你是一位醫療 AI 合規審查專家。原則：{item['title']}，定義：{item['desc']}。
    歷史參考：{rag_history}。內容：{context_text[:12000]}。
    以 JSON 回覆: {{"status": "存在/不存在", "summary": "摘要", "suggestion": "建議"}}"""
    try:
        response = model.generate_content(prompt)
        clean_text = re.sub(r"```json\n?|\n?```", "", response.text).strip()
        return json.loads(clean_text)
    except:
        return {"status": "檢核錯誤", "summary": "API 錯誤", "suggestion": ""}

def get_embedding(text):
    try:
        result = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")
        return result['embedding']
    except: return [0] * 768

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def run_full_analysis(full_text, rag_df):
    """執行單份文件的完整分析"""
    all_items = TRANSPARENCY_9 + GOVERNANCE_2
    results_t, results_g = [], []
    pdf_vec = get_embedding(full_text[:2000])

    for i, item in enumerate(all_items):
        history = ""
        if not rag_df.empty:
            rel_rows = rag_df[rag_df["Principle"] == item["title"]].copy()
            if not rel_rows.empty:
                # 簡單取最後三筆作為歷史參考（或依相似度）
                history = "\n".join([f"- {fb}" for fb in rel_rows["UserFeedback"].tail(3).tolist()])
        
        res = analyze_item(item, full_text, rag_history=history)
        if i < 9: results_t.append(res)
        else: results_g.append(res)
    return {"t": results_t, "g": results_g}

def convert_all_to_xlsx(batch_results):
    """將所有檔案結果轉為多分頁 XLSX"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for filename, res in batch_results.items():
            data = []
            for i, item in enumerate(res['t']):
                data.append({"分類": "九大透明性", "項目": TRANSPARENCY_9[i]['title'], "狀態": item['status'], "摘要": item['summary'], "建議": item['suggestion']})
            for i, item in enumerate(res['g']):
                data.append({"分類": "核心治理", "項目": GOVERNANCE_2[i]['title'], "狀態": item['status'], "摘要": item['summary'], "建議": item['suggestion']})
            
            df = pd.DataFrame(data)
            # Sheet 名稱不能超過 31 字元，且不能有特殊字元
            sheet_name = re.sub(r'[\\/*?:\[\]]', '', filename)[:30]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()

# ---------- 4. UI 介面 ----------

def main():
    st.set_page_config(page_title="醫療 AI 批次檢核", layout="wide")
    st.title("🛡️ 負責任 AI 批次檢核系統")

    if 'batch_results' not in st.session_state:
        st.session_state['batch_results'] = {}

    with st.sidebar:
        st.header("1. 檔案上傳")
        pdf_files = st.file_uploader("上傳多份計畫書 PDF", type="pdf", accept_multiple_files=True)
        btn = st.button("🚀 開始批次分析", use_container_width=True)

    if pdf_files and btn:
        rag_df = get_rag_df_from_github()
        new_results = {}
        
        progress_bar = st.progress(0)
        for idx, file in enumerate(pdf_files):
            with st.status(f"正在分析 ({idx+1}/{len(pdf_files)}): {file.name}"):
                doc = fitz.open(stream=file.read(), filetype="pdf")
                text = "\n".join([page.get_text() for page in doc])
                new_results[file.name] = run_full_analysis(text, rag_df)
            progress_bar.progress((idx + 1) / len(pdf_files))
        
        st.session_state['batch_results'] = new_results
        st.success("全部分析完成！")

    # 顯示結果與下載
    if st.session_state['batch_results']:
        results = st.session_state['batch_results']
        
        col1, col2 = st.columns([1, 4])
        with col1:
            xlsx_data = convert_all_to_xlsx(results)
            st.download_button(
                label="📥 下載多分頁 Excel 報告",
                data=xlsx_data,
                file_name=f"批次檢核報告_{datetime.datetime.now().strftime('%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # 預覽部分內容
        selected_file = st.selectbox("查看預覽結果：", list(results.keys()))
        if selected_file:
            res = results[selected_file]
            tab1, tab2 = st.tabs(["透明性原則", "治理指標"])
            with tab1:
                for i, r in enumerate(res['t']):
                    with st.expander(f"{i+1}. {TRANSPARENCY_9[i]['title']} - {r['status']}"):
                        st.write(f"**摘要:** {r['summary']}")
                        st.write(f"**建議:** {r['suggestion']}")
            with tab2:
                st.table(pd.DataFrame(res['g']))

if __name__ == "__main__":
    main()

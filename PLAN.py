import os
import json
import re
import base64
import datetime
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
import requests
from io import StringIO, BytesIO # 新增 BytesIO 用於 Excel 匯出
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
    model_name="gemini-2.5-pro", 
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.1,
    },
    safety_settings=SAFETY_SETTINGS
)
# ---------- 2. 原則定義 (保持不變) ----------
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
    {"title": "可解釋性分析", "desc": "詳述模型決策依據是否導入SHAP、Grad-CAM、LIME、Saliency Map、Attention Map等，輔助醫師驗證輸出結果。"},
    {"title": "AI生命週期管理", "desc": "系統是否具定期監測與性能評估計畫，以適應醫療數據漂移並涵蓋維護、更新與模型退役之程序。"}
]
# ---------- 3. 功能函式 ----------
import csv
from google.cloud import vision
def perform_google_ocr(pdf_file_bytes):
    """使用 Google Cloud Vision API 進行掃描檔辨識 (從 Streamlit Secrets 讀取憑證)"""
    from google.oauth2 import service_account
    
    # --- 修改處：從 st.secrets 取得憑證資訊 ---
    try:
        if "gcp_service_account" in st.secrets:
            creds_info = st.secrets["gcp_service_account"]
            credentials = service_account.Credentials.from_service_account_info(creds_info)
            client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            st.error("找不到名為 'gcp_service_account' 的 Secrets 設定。")
            return ""
    except Exception as e:
        st.error(f"初始化 Google Cloud Vision 失敗: {e}")
        return ""
    # ---------------------------------------

    doc = fitz.open(stream=pdf_file_bytes, filetype="pdf")
    full_text = ""
    
    for page in doc:
        # 將頁面渲染為圖片 (2.0 矩陣約為 144 DPI，若要更精準可用 3, 3)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        
        image = vision.Image(content=img_bytes)
        response = client.text_detection(image=image)
        
        if response.text_annotations:
            # text_annotations[0] 包含該頁面的完整辨識文字
            full_text += response.text_annotations[0].description + "\n"
            
    return full_text

def get_smart_text(pdf_file_bytes):
    """智慧判斷：若原生文字太少則啟動 OCR"""
    doc = fitz.open(stream=pdf_file_bytes, filetype="pdf")
    native_text = "\n".join([page.get_text() for page in doc])
    
    # 如果原生文字長度過短 (例如 < 100字)，判定為掃描檔
    if len(native_text.strip()) < 100:
        return perform_google_ocr(pdf_file_bytes)
    return native_text

def get_rag_df_from_github():
    """從 GitHub 讀取目前的 RAG 庫，增加強健性處理"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}", 
        "Accept": "application/vnd.github.v3+json",
        "Cache-Control": "no-cache"
    }
    
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            file_json = res.json()
            # 使用 utf-8 讀取並過濾掉可能導致問題的 null bytes
            content = base64.b64decode(file_json['content']).decode('utf-8').replace('\x00', '')
            
            if not content.strip(): 
                return pd.DataFrame(columns=["Principle", "UserFeedback"])
            
            # 核心修正：使用更嚴格的解析參數
            return pd.read_csv(
                StringIO(content), 
                sep=',',
                quotechar='"',          # 指定引號字元
                doublequote=True,       # 允許雙引號轉義
                escapechar='\\',        # 增加逃逸字元處理
                on_bad_lines='skip',    # 遇到毀損行直接跳過，確保程式不崩潰
                engine='python'         # Python 引擎對損壞檔案的容忍度較高
            )
        else:
            return pd.DataFrame(columns=["Principle", "UserFeedback"])
            
    except Exception as e:
        # 如果發生 ParserError，回傳空表讓程式繼續執行，並提示用戶
        st.warning(f"RAG 庫格式受損，已自動重置。錯誤詳情: {e}")
        return pd.DataFrame(columns=["Principle", "UserFeedback"])

def update_rag_to_github(principle, feedback):
    """將回饋存入 GitHub，強制執行格式標準化"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    # 1. 取得現有資料 (已包含錯誤處理)
    df = get_rag_df_from_github()
    
    # 再次確認 SHA
    res = requests.get(url, headers=headers)
    sha = res.json().get('sha') if res.status_code == 200 else None

    # 2. 清理 Feedback 內容：移除所有可能干擾 CSV 的字元
    # 移除雙引號避免嵌套錯誤，並統一換行符號
    clean_feedback = str(feedback).replace('"', "'").replace('\n', ' ').strip()
    # 3. 加入新列
    new_data = pd.DataFrame([{
        "Principle": principle,
        "UserFeedback": clean_feedback,
    }])
    
    df = pd.concat([df, new_data], ignore_index=True)

    # 4. 轉回 CSV：強制對所有欄位加引號，確保解析安全
    csv_content = df.to_csv(
        index=False, 
        encoding='utf-8', 
        quoting=csv.QUOTE_ALL, 
        quotechar='"'
    )
    
    encoded_content = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
    
    payload = {
        "message": f"Fix & Update RAG feedback for {principle}",
        "content": encoded_content,
        "sha": sha
    }
    
    put_res = requests.put(url, headers=headers, json=payload)
    return put_res.status_code in [200, 201]


def generalize_feedback(specific_feedback):
    prompt = f"""
    使用者針對醫療 AI 審查提供了具體修正建議：'{specific_feedback}'
    請將其以簡短純文字轉為通用的審查原則，使其能適用於其他不同的計畫書或不同任務模型。
    只回傳轉化後的文字，不要有其他解釋。
    """
    response = model.generate_content(prompt, generation_config={"response_mime_type": "text/plain"})
    return response.text.strip()     





def analyze_item(item, context_text, rag_history=""):
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
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        return [0] * 768  



def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    
def run_full_analysis(full_text):
    rag_df = get_rag_df_from_github()
    all_items = TRANSPARENCY_9 + GOVERNANCE_2
    results_t = []
    results_g = []
    pdf_context = full_text[:2000]

    try:
        pdf_vec = get_embedding(pdf_context)
    except:
        pdf_vec = None

    for i, item in enumerate(all_items):
        history = ""
        if not rag_df.empty and "UserFeedback" in rag_df.columns:
            rel_rows = rag_df[rag_df["Principle"] == item["title"]].copy()
            if not rel_rows.empty and pdf_vec is not None:
                similarities = []
                for fb in rel_rows["UserFeedback"].tolist():
                    try:
                        fb_vec = get_embedding(fb)
                        sim = cosine_similarity(pdf_vec, fb_vec)
                        similarities.append(sim)
                    except:
                        similarities.append(0)
                rel_rows["sim"] = similarities
                top_3 = rel_rows.sort_values(by="sim", ascending=False).head(3)
                history = "\n".join([f"- {row['UserFeedback']}" for _, row in top_3.iterrows()])

        res = analyze_item(item, full_text, rag_history=history)
        if i < 9:
            results_t.append(res)
        else:
            results_g.append(res)
    return {"t": results_t, "g": results_g}


# --- 修改處：支援多頁簽 Excel 匯出 ---
def convert_all_results_to_xlsx():
    """將所有 PDF 的分析結果轉換為多頁簽 Excel"""
    if 'batch_results' not in st.session_state or not st.session_state['batch_results']:
        return None
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for filename, res in st.session_state['batch_results'].items():
            data = []
            # 處理 9 大原則
            for i, item in enumerate(res['t']):
                data.append({
                    "分類": "九大透明性原則",
                    "項目": TRANSPARENCY_9[i]['title'],
                    "狀態": item['status'],
                    "摘要": item['summary'],
                    "建議": item['suggestion']
                })
            # 處理 2 大指標
            for i, item in enumerate(res['g']):
                data.append({
                    "分類": "核心治理指標",
                    "項目": GOVERNANCE_2[i]['title'],
                    "狀態": item['status'],
                    "摘要": item['summary'],
                    "建議": item['suggestion']
                })
            
            df = pd.DataFrame(data)
            # Excel 頁簽名稱限制 31 字元且不可有特殊符號
            sheet_name = re.sub(r'[\\/*?:\[\]]', '', filename)[:30]
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    
    return output.getvalue()


# ---------- 4. UI 介面 ----------
def main():
    st.set_page_config(page_title="醫療 AI 治理檢核", layout="wide")
    st.title("🛡️ 負責任 AI 自動檢核系統")
    
    # 1. 初始化 session_state
    if 'batch_results' not in st.session_state:
        st.session_state['batch_results'] = {}
    if 'analysis_done' not in st.session_state:
        st.session_state['analysis_done'] = False

    with st.sidebar:
        st.header("1. 檔案讀取")
        # 確保第一個參數是標題字串，並加上 unique key
        pdf_files = st.file_uploader(
            "上傳多份計畫書 PDF", 
            type="pdf", 
            accept_multiple_files=True)
        # 為按鈕也加上 key 確保唯一性
        btn = st.button("🚀 開始批次分析", use_container_width=True, key="start_analysis_btn")
        st.divider()

    # 2. 執行分析邏輯 (僅在點擊按鈕時觸發)
    if pdf_files and btn:
        st.session_state['batch_results'] = {} 
        st.session_state['analysis_done'] = False
        progress_bar = st.progress(0)
        for idx, pdf_file in enumerate(pdf_files):
            # 使用 status 顯示當前進度，完成後它會停留在頁面直到下一次互動
            with st.status(f"正在分析 ({idx+1}/{len(pdf_files)}): {pdf_file.name}...", expanded=True) as status:
                log_placeholder = st.empty() # 在 status 內部建立佔位空間
    
                log_placeholder.write("📂 正在讀取 PDF 內容...")
                doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                full_text = "\n".join([page.get_text() for page in doc])
    
                log_placeholder.write("🧠 正在呼叫 Gemini Pro 進行合規性審查...")
                results = run_full_analysis(full_text)
    
                log_placeholder.write("📊 正在彙整分析結果...")
                st.session_state['batch_results'][pdf_file.name] = results
    
                # 這裡可以選擇清除 log 或保留
                log_placeholder.empty() 
                status.update(label=f"✅ {pdf_file.name} 分析完成", state="complete", expanded=False)
                progress_bar.progress((idx + 1) / len(pdf_files))
        
        st.session_state['analysis_done'] = True
        # 這裡不需要 rerun，Streamlit 偵測到 state 改變會自動更新

    # 3. 顯示結果 (這部分保持你原本的顯示邏輯)
    if st.session_state['batch_results']:
        file_names = list(st.session_state['batch_results'].keys())
        
        # 新增：在報表上方也提供一個明顯的下載按鈕
        c1, c2 = st.columns([8, 2])
        with c2:
            xlsx_data_main = convert_all_results_to_xlsx()
            st.download_button(
                label="📥 下載所有分析報表",
                data=xlsx_data_main,
                file_name=f"Medical_AI_Audit_{datetime.datetime.now().strftime('%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="main_download"
            )
        tabs = st.tabs(file_names)
        
        for i, name in enumerate(file_names):
            with tabs[i]:
                res_data = st.session_state['batch_results'][name]
                st.subheader(f"📊 {name} - 九大透明性原則")
                t_data = res_data['t']
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
                g_data = res_data['g']
                df_g = pd.DataFrame([{
                    "評估項目": GOVERNANCE_2[i]['title'],
                    "狀態": d['status'],
                    "摘要": d['summary'],
                    "建議": d['suggestion']
                } for i, d in enumerate(g_data)])
                st.table(df_g)
        # RAG 回饋區域 (針對最後一個或選定的檔案)
        st.divider()
        st.subheader("📝 訓練 AI 的判斷經驗 (RAG)")
        with st.form("rag_feedback_form"):
            all_titles = [i['title'] for i in (TRANSPARENCY_9 + GOVERNANCE_2)]
            col1, col2 = st.columns([1, 2])
            with col1:
                selected_title = st.selectbox("選擇要修正的項目", all_titles)
            with col2:
                user_comment = st.text_area("修正建議", placeholder="例如：應加強對表格內數據的識別...")
            submit_rag = st.form_submit_button("✅ 送出經驗並優化未來分析")
            if submit_rag and user_comment:
                generalized_comment = generalize_feedback(user_comment)
                with st.spinner("同步至 GitHub 中..."):
                    if update_rag_to_github(selected_title, generalized_comment):
                        st.success("回饋成功！")
                    else:
                        st.error("寫入失敗。")

if __name__ == "__main__":
    main()

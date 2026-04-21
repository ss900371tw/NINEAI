import os
import json
import re
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

# ---------- 1. 初始化與環境設定 ----------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("請在 .env 檔案中設定 GOOGLE_API_KEY")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# 設定安全性：放寬醫療專有名詞的限制，確保解析不中斷
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# 初始化 Gemini 2.5 Pro
model = genai.GenerativeModel(
    model_name="gemini-2.5-pro", # 根據您的 API 權限設定
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.2,  # 降低隨機性，提高分析穩定度
    },
    safety_settings=SAFETY_SETTINGS
)

# ---------- 2. 原則定義 ----------
TRANSPARENCY_9 = [
    {"title": "介入詳情及輸出", "desc": "詳細說明 AI 產品的技術規格，包含模型架構、輸入數據的格式要求（如影像解析度、採樣頻率）以及輸出的具體臨床含義。"},
    {"title": "介入目的", "desc": "明確界定該 AI 在臨床工作流中的角色，包含預期用途（Intended Use, IU）與適應症（Indications for Use, IFU）。"},
    {"title": "警告範圍外使用", "desc": "列出該產品的禁忌症（Contraindications）與技術極限，即在何種情況下 AI 可能失效或產生錯誤誤導。"},
    {"title": "開發詳情及輸入", "desc": "揭露訓練資料集的特徵，包括數據來源、入選與排除標準、資料分布情形及標註（Labeling）流程。"},
    {"title": "開發公平性過程", "desc": "說明開發團隊如何識別並緩解潛在的演算法偏差（Bias），確保模型對不同群體表現一致。"},
    {"title": "外部驗證過程", "desc": "使用未參與開發過程的獨立資料集進行效能測試，以驗證模型的泛化能力。"},
    {"title": "表現量化指標", "desc": "提供多維度的效能評估報告，而非單一的準確率。"},
    {"title": "實施與持續維護", "desc": "描述產品部署後的監控機制，包含系統整合需求、使用者教育訓練及異常回報流程。"},
    {"title": "更新與公平性評估", "desc": "規範模型版本迭代的流程，確保在軟體更新或重新訓練後，模型的效能與安全性不會倒退。"}
]

GOVERNANCE_2 = [
    {"title": "可解釋性分析", "desc": "利用技術讓醫師理解模型決策，確保輸出可驗證。"},
    {"title": "AI生命週期管理", "desc": "從開發到部署的全程風險評估與合規性監測。"}
]

# ---------- 3. 核心邏輯 ----------

def analyze_item(item, context_text):
    """呼叫 Gemini 2.5 Pro 進行單項分析"""
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
        # 清除 Markdown 標籤以防解析錯誤
        clean_text = re.sub(r"```json\n?|\n?```", "", response.text).strip()
        return json.loads(clean_text)
    except Exception as e:
        return {"status": "檢核錯誤", "summary": f"API 錯誤: {str(e)}", "suggestion": ""}

def run_full_analysis(full_text):
    all_items = TRANSPARENCY_9 + GOVERNANCE_2
    # 2.5 Pro 支援高併發，這裡開 6 個 Thread 同時跑
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

    if pdf_file and btn:
        with st.spinner("Gemini 2.5 Pro 正在分析中..."):
            # 讀取 PDF 文字
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            full_text = "\n".join([page.get_text() for page in doc])
            
            # 分析
            results = run_full_analysis(full_text)
            st.session_state['res_t'] = results['t']
            st.session_state['res_g'] = results['g']

    # --- 顯示結果 ---
    if 'res_t' in st.session_state:
        st.subheader("📊 九大透明性原則 (九宮格)")
        t_data = st.session_state['res_t']
        
        # 3x3 呈現
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

if __name__ == "__main__":
    main()

import os
import json
import re
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

# 向量資料庫相關
import chromadb
from chromadb.utils import embedding_functions

# ---------- 1. 初始化與環境設定 ----------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("請在 .env 檔案中設定 GOOGLE_API_KEY")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# 初始化 ChromaDB (本地持久化存儲)
CHROMA_DATA_PATH = "chroma_db_medical_ai"
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# 使用 Gemini 的 Embedding 模型來將文字向量化
gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=GOOGLE_API_KEY,
    model_name="models/text-embedding-004"
)

# 取得或創建向量集
collection = chroma_client.get_or_create_collection(
    name="compliance_feedback",
    embedding_function=gemini_ef
)

# 安全性設定
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# 初始化 Gemini 模型
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro", # 建議使用 pro 等級處理複雜邏輯
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.1,
    },
    safety_settings=SAFETY_SETTINGS
)

# ---------- 2. 原則定義 (保持不變) ----------
TRANSPARENCY_9 = [
    {"title": "介入詳情及輸出", "desc": "詳細說明 AI 產品的技術規格，包含模型架構、輸入數據的要求與輸出的具體臨床含義。"},
    {"title": "介入目的", "desc": "明確界定 AI 在臨床工作流中的角色，包含預期用途（IU）與適應症（IFU）。"},
    {"title": "警告範圍外使用", "desc": "列出產品的禁忌症（Contraindications）與技術極限。"},
    {"title": "開發詳情及輸入", "desc": "揭露訓練資料集的特徵，包括來源、入選標準、資料分布及標註流程。"},
    {"title": "開發公平性過程", "desc": "說明團隊如何識別並緩解潛在的演算法偏差（Bias）。"},
    {"title": "外部驗證過程", "desc": "使用獨立資料集進行效能測試，驗證模型的泛化能力。"},
    {"title": "表現量化指標", "desc": "提供多維度的效能評估報告（AUC, Sensitivity, Specificity等）。"},
    {"title": "實施與持續維護", "desc": "描述部署後的監控機制、使用者訓練及異常回報流程。"},
    {"title": "更新與公平性評估", "desc": "規範模型版本迭代流程，確保更新後效能不倒退。"}
]

GOVERNANCE_2 = [
    {"title": "可解釋性分析", "desc": "利用技術讓醫師理解模型決策（如 Grad-CAM），確保輸出可驗證。"},
    {"title": "AI生命週期管理", "desc": "從開發到部署的全程風險評估與合規性監測。"}
]

# ---------- 3. RAG 核心邏輯 ----------

def get_rag_context(item_title, n_results=2):
    """從 ChromaDB 檢索與當前原則最相關的歷史建議"""
    try:
        results = collection.query(
            query_texts=[f"關於 {item_title} 的專家審查建議"],
            n_results=n_results
        )
        if results['documents'] and len(results['documents'][0]) > 0:
            formatted_history = "\n".join([f"- 參考歷史建議: {doc}" for doc in results['documents'][0]])
            return f"\n【參考過去專家回饋】\n{formatted_history}"
    except Exception:
        pass
    return ""

def add_feedback_to_db(principle, user_comment, ai_summary):
    """將專家回饋存入向量庫"""
    doc_content = f"項目：{principle} | 專家建議：{user_comment} | 原始AI摘要：{ai_summary}"
    doc_id = f"id_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    collection.add(
        documents=[doc_content],
        metadatas=[{"principle": principle}],
        ids=[doc_id]
    )

def analyze_item(item, context_text):
    # 執行檢索：撈取歷史經驗
    history = get_rag_context(item['title'])
    
    prompt = f"""
    你是一位醫療 AI 合規性審查專家。請針對以下原則分析文件內容：
    原則：{item['title']}
    定義：{item['desc']}
    
    {history}
    
    待審核文件內容：{context_text[:12000]}
    
    請依據文件內容，並『特別參考』歷史建議中提到的標準，以 JSON 格式回覆：
    {{
      "status": "存在" 或 "不存在",
      "summary": "具體做法摘要（若不存在則寫未見描述）",
      "suggestion": "缺失建議（若已存在則留空）"
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
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda x: analyze_item(x, full_text), all_items))
    return {"t": results[:9], "g": results[9:]}

# ---------- 4. Streamlit UI ----------

def main():
    st.set_page_config(page_title="醫療 AI 治理檢核 (RAG版)", layout="wide")
    st.title("🛡️ 負責任 AI 自動檢核系統 (RAG-Enabled)")

    # 側邊欄：上傳與觸發
    with st.sidebar:
        st.header("1. 檔案讀取")
        pdf_file = st.file_uploader("上傳計畫書 PDF", type="pdf")
        btn = st.button("🚀 開始分析", use_container_width=True)
        
        st.divider()
        if st.button("🧹 清空資料庫記錄"):
            # 危險操作：僅供測試時使用
            st.warning("這將清除所有已存的 RAG 經驗資料。")
            # 實際執行需要 delete collection 的代碼

    # 主要分析邏輯
    if pdf_file and btn:
        with st.spinner("Gemini 正在結合歷史經驗進行分析..."):
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            full_text = "\n".join([page.get_text() for page in doc])
            
            results = run_full_analysis(full_text)
            st.session_state['res_t'] = results['t']
            st.session_state['res_g'] = results['g']

    # 顯示分析結果
    if 'res_t' in st.session_state:
        st.subheader("📊 九大透明性原則檢核")
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

        # ---------- 新增：專家回饋入口 ----------
        st.divider()
        st.subheader("💬 專家優化回饋 (將存入 RAG 知識庫)")
        with st.form("feedback_form", clear_on_submit=True):
            target_item = st.selectbox("請選擇要提供建議的項目：", 
                                       [p['title'] for p in (TRANSPARENCY_9 + GOVERNANCE_2)])
            user_comment = st.text_area("請輸入您對此項目的專業修正建議或理想描述標準：")
            submitted = st.form_submit_button("送出建議並優化 AI")
            
            if submitted and user_comment:
                # 找到對應的 AI 分析摘要
                all_results = st.session_state['res_t'] + st.session_state['res_g']
                all_titles = [p['title'] for p in (TRANSPARENCY_9 + GOVERNANCE_2)]
                current_summary = all_results[all_titles.index(target_item)]['summary']
                
                # 存入 ChromaDB
                add_feedback_to_db(target_item, user_comment, current_summary)
                st.success(f"成功！下次分析『{target_item}』時，我將參考這條建議。")

if __name__ == "__main__":
    main()

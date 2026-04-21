import os
import json
import re
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

# 向量資料庫 Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ---------- 1. 初始化與環境設定 ----------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Qdrant 設定 (建議改用環境變數)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost") 
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

if not GOOGLE_API_KEY:
    st.error("請在 .env 檔案中設定 GOOGLE_API_KEY")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# 初始化 Qdrant Client (可連至遠端伺服器)
qdrant_client = QdrantClient(host=QDRANT_HOST, api_key=QDRANT_API_KEY, port=6333)
COLLECTION_NAME = "compliance_feedback"

# 確保 Collection 存在 (Gemini text-embedding-001 維度為 768)
try:
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )
except Exception as e:
    st.error(f"Qdrant 連線失敗: {e}")

# 安全性設定與模型初始化
SAFETY_SETTINGS = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}] # 簡化示範
model = genai.GenerativeModel(
    model_name="gemini-2.5-pro", 
    generation_config={"response_mime_type": "application/json", "temperature": 0.1},
    safety_settings=SAFETY_SETTINGS
)

# ---------- 2. 原則定義 (與原文一致) ----------
TRANSPARENCY_9 = [...] # 此處省略，請沿用您原本的清單
GOVERNANCE_2 = [...]   # 此處省略，請沿用您原本的清單

# ---------- 3. RAG 核心邏輯 (Qdrant 版) ----------

def get_embedding(text):
    """將文字轉換為向量"""
    result = genai.embed_content(
        model="models/text-embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return result['embedding']

def get_rag_context(item_title, n_results=2):
    """從 Qdrant 檢索歷史專家審查建議"""
    try:
        query_vector = get_embedding(f"關於 {item_title} 的專家審查建議")
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=n_results
        )
        if search_result:
            formatted_history = "\n".join([f"- 歷史修正建議: {hit.payload['user_comment']}" for hit in search_result])
            return f"\n【參考過去專家回饋（Qdrant RAG 增強）】\n{formatted_history}"
    except Exception as e:
        print(f"RAG Retrieval Error: {e}")
    return ""

def add_feedback_to_db(principle, user_comment, ai_summary):
    """將專家回饋存入 Qdrant 雲端/伺服器端"""
    doc_content = f"項目：{principle} | 建議：{user_comment}"
    vector = get_embedding(doc_content)
    
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=str(pd.Timestamp.now().timestamp()).replace(".", ""),
                vector=vector,
                payload={
                    "principle": principle,
                    "user_comment": user_comment,
                    "ai_summary": ai_summary,
                    "timestamp": str(pd.Timestamp.now())
                }
            )
        ]
    )

def analyze_item(item, context_text):
    history = get_rag_context(item['title'])
    prompt = f"""
    你是一位醫療 AI 合規性審查專家。請針對以下原則分析文件內容：
    原則：{item['title']}
    定義：{item['desc']}
    {history}
    待審核計畫書內容：{context_text[:12000]}
    請依據計畫書，優先參考歷史建議，以 JSON 回覆：
    {{ "status": "存在/不存在", "summary": "摘要", "suggestion": "建議" }}
    """
    try:
        response = model.generate_content(prompt)
        clean_text = re.sub(r"```json\n?|\n?```", "", response.text).strip()
        return json.loads(clean_text)
    except Exception as e:
        return {"status": "錯誤", "summary": str(e), "suggestion": ""}

def run_full_analysis(full_text):
    all_items = TRANSPARENCY_9 + GOVERNANCE_2
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(lambda x: analyze_item(x, full_text), all_items))
    return {"t": results[:9], "g": results[9:]}


# ---------- 4. Streamlit UI ----------

def main():
    st.set_page_config(page_title="醫療 AI 治理檢核 (Gemini 2.5 Pro + Qdrant)", layout="wide")
    
    # 自定義 CSS 讓介面更專業
    st.markdown("""
        <style>
        .stAlert { margin-top: 10px; }
        .main-title { font-size: 2.2rem; font-weight: 700; color: #1E88E5; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-title">🛡️ 負責任 AI 自動檢核系統 (Shared Knowledge)</p>', unsafe_allow_html=True)

    # ---------- 側邊欄：檔案與設定 ----------
    with st.sidebar:
        st.header("1. 檔案管理")
        pdf_file = st.file_uploader("上傳計畫書 PDF", type="pdf")
        
        st.divider()
        st.header("2. 執行分析")
        analyze_btn = st.button("🚀 開始 Qdrant RAG 增強分析", use_container_width=True)
        
        st.divider()
        st.status(f"連結至 Qdrant: `{QDRANT_HOST}`")
        st.info("當前版本：Gemini 2.5 Pro + Qdrant Distributed RAG")

    # ---------- 主區塊：分析邏輯 ----------
    if pdf_file and analyze_btn:
        with st.spinner("Gemini 2.5 Pro 正在從 Qdrant 檢索歷史經驗並審查計畫書..."):
            try:
                # 讀取 PDF
                doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                full_text = "\n".join([page.get_text() for page in doc])
                
                # 執行 6 線程並行分析
                results = run_full_analysis(full_text)
                
                # 將結果存入 session_state
                st.session_state['res_t'] = results['t']
                st.session_state['res_g'] = results['g']
                st.session_state['analysis_done'] = True
            except Exception as e:
                st.error(f"分析過程中發生錯誤: {e}")

    # ---------- 結果展示區 ----------
    if st.session_state.get('analysis_done'):
        # 1. 透明性原則顯示 (3x3 網格)
        st.subheader("📊 九大透明性原則檢核")
        t_data = st.session_state['res_t']
        
        for r in range(3):
            cols = st.columns(3)
            for c in range(3):
                idx = r * 3 + c
                if idx < len(t_data):
                    item = t_data[idx]
                    with cols[c]:
                        status_color = "green" if "存在" in item['status'] else "red"
                        with st.expander(f"{idx+1}. {TRANSPARENCY_9[idx]['title']}", expanded=True):
                            st.markdown(f"**狀態：** :{status_color}[{item['status']}]")
                            st.info(f"**摘要：**\n{item['summary']}")
                            if item['suggestion']:
                                st.warning(f"💡 **建議：**\n{item['suggestion']}")

        st.divider()

        # 2. 核心治理指標顯示 (表格)
        st.subheader("📋 核心治理指標")
        g_data = st.session_state['res_g']
        df_g = pd.DataFrame([{
            "評估項目": GOVERNANCE_2[i]['title'],
            "狀態": d['status'],
            "摘要": d['summary'],
            "建議": d['suggestion']
        } for i, d in enumerate(g_data)])
        st.table(df_g)

        # 3. 專家回饋入口 (關鍵功能)
        st.divider()
        st.subheader("💬 專家優化回饋 (訓練共享知識庫)")
        st.markdown("如果您不認同 AI 的判定，請輸入您的建議。這些建議將會被存入 Qdrant，並在**下次任何人**進行分析時，成為 Gemini 的參考標準。")
        
        with st.form("feedback_form", clear_on_submit=True):
            all_titles = [p['title'] for p in (TRANSPARENCY_9 + GOVERNANCE_2)]
            target_item = st.selectbox("選擇要指正的項目：", all_titles)
            user_comment = st.text_area("您的專業建議 (例如：應補充說明數據去識別化之具體流程)")
            
            submitted = st.form_submit_button("📢 送出建議並同步至 Qdrant")
            
            if submitted:
                if user_comment.strip():
                    # 取得目前該項目的 AI 摘要作為 Context 存入
                    all_res = st.session_state['res_t'] + st.session_state['res_g']
                    current_idx = all_titles.index(target_item)
                    current_ai_summary = all_res[current_idx]['summary']
                    
                    try:
                        add_feedback_to_db(target_item, user_comment, current_ai_summary)
                        st.success(f"✅ 成功！您的建議已存入 Qdrant。未來 Gemini 將學習到：『{user_comment[:30]}...』")
                    except Exception as e:
                        st.error(f"存入 Qdrant 失敗: {e}")
                else:
                    st.warning("請輸入建議內容後再送出。")

    else:
        # 初始畫面提示
        st.info("請於側邊欄上傳醫療 AI 計畫書 PDF 並點擊開始分析。")

if __name__ == "__main__":
    # 初始化 session_state 防止重整遺失資料
    if 'analysis_done' not in st.session_state:
        st.session_state['analysis_done'] = False
    main()

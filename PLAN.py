import os
import re
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

# ---------- 初始化 ----------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)

# 建議使用 gemini-1.5-flash 或 gemini-2.0-flash 以獲得最穩定的輸出
model = genai.GenerativeModel("gemini-1.5-flash") 

# FAISS 向量庫初始化
INDEX_FILE_PATH = "faiss_index"
vector_store = None
try:
    vector_store = FAISS.load_local(INDEX_FILE_PATH, HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
except Exception as e:
    print("⚠️ 無法載入 FAISS 向量庫：", e)

# ---------- 九大透明性原則定義 ----------
TRANSPARENCY_PRINCIPLES = [
    "介入詳情及輸出：說明人工智慧模型的基本特徵（如模型架構、訓練技術）以及模型輸出的形式。",
    "介入目的：說明人工智慧模型設計的核心目標以及適用情境。",
    "介入的警告範圍外使用：說明人工智慧模型適用、不適用範圍，及其可能發生之風險。",
    "介入開發詳情及輸入特徵：說明 AI 是如何被開發出來的，以及 AI 在訓練和運作時是使用哪些數據特徵",
    "確保介入開發公平性的過程：說明開發AI系統的過程中，採取哪些具體方法防止或減輕模型在不同群體、用戶或情況下產生偏見或不公平結果",
    "外部驗證過程：說明如何將開發完成的AI系統帶到真實或模擬的、與開發環境不同的場所或數據集上進行測試和驗證，以確認其穩定性、準確性和泛化能力",
    "模型表現的量化指標：說明此人工智慧模型的量化評估指標，如模型的準確率、模型的召回率、模型的F1分數、模型的AUC曲線",
    "介入實施和使用的持續維護：說明AI部署 and 實際使用之後，如何進行後續保養、監控、修復錯誤、處理性能衰退及規劃未來版本更新等工作",
    "更新和持續驗證或公平性評估計劃：說明如何定期對模型重新訓練、功能升級，並透過持續監測和評估系統公平性進行調整，確保其不對特定群體產生偏見以符合臨床需求。"
]

# ---------- 輔助函式 ----------
def inject_custom_css():
    st.markdown("""
    <style>
    .flip-card {
      background-color: transparent;
      width: 100%;
      height: 320px; /* 增加高度讓內容更好讀 */
      perspective: 1000px;
      margin-bottom: 25px;
    }
    .flip-card-inner {
      position: relative;
      width: 100%;
      height: 100%;
      text-align: center;
      transition: transform 0.6s;
      transform-style: preserve-3d;
      cursor: pointer;
    }
    .flip-card:hover .flip-card-inner {
      transform: rotateY(180deg);
    }
    .flip-card-front, .flip-card-back {
      position: absolute;
      width: 100%;
      height: 100%;
      -webkit-backface-visibility: hidden;
      backface-visibility: hidden;
      display: flex;
      flex-direction: column;
      padding: 20px;
      border-radius: 12px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.1);
      box-sizing: border-box;
    }
    .flip-card-front {
      background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
      color: white;
      justify-content: center;
      align-items: center;
    }
    .flip-card-back {
      background-color: #ffffff;
      color: #2c3e50;
      transform: rotateY(180deg);
      border: 1px solid #e0e0e0;
      justify-content: flex-start; 
      overflow-y: auto;
    }
    .flip-card-back::-webkit-scrollbar { width: 4px; }
    .flip-card-back::-webkit-scrollbar-thumb { background: #cbd5e0; border-radius: 10px; }
    .status-badge { margin-top: 10px; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: bold; color: white; }
    .summary-text { font-size: 0.9em; text-align: left; line-height: 1.5; margin-bottom: 10px; }
    .suggestion-box {
      padding: 8px;
      background-color: #fff5f5;
      border-left: 4px solid #f56565;
      font-size: 0.8em;
      text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

def extract_text_by_line(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    lines = []
    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if text: lines.append(text)
    return "\n\n".join(lines)

def get_gemini_response(prompt):
    try:
        response = model.generate_content(prompt)
        # 額外處理，避免 AI 在內容中夾雜 markdown 語法導致 HTML 破裂
        return response.text.replace("```html", "").replace("```", "").strip()
    except Exception as e:
        return f"⚠️ 錯誤：{e}"

def gen_missing_suggestion(principle_text):
    prompt = f"你是一位專業 AI 模型透明性報告撰寫員。若文件未涵蓋下列原則，請寫出建議補上的內容：\n\n{principle_text}\n\n請用正式繁體中文撰寫，不要使用 Markdown 標籤。"
    return get_gemini_response(prompt)

def build_transparency_prompts(principles, full_text, rag_docs_k=3):
    prompts = []
    rag_context = ""
    if vector_store and rag_docs_k > 0:
        docs = vector_store.similarity_search("AI Transparency", k=rag_docs_k)
        rag_context = "\n---\n".join(doc.page_content for doc in docs)
    for p in principles:
        title = p.split('：', 1)[0]
        desc = p.split('：', 1)[1]
        prompt = f"---- 透明性原則 ----\n項目：{title}\n要求：{desc}\n\n---- 文件內容 ----\n{full_text[:4000]}\n\n---- 參考背景 ----\n{rag_context}\n\n---- 回覆格式 ----\n狀態: 存在 / 不存在\n摘要: (摘要說明)"
        prompts.append(prompt)
    return prompts

def parse_transparency_response(response_text):
    status = "不存在" if "不存在" in response_text else "存在"
    summary = "未見相關描述"
    m = re.search(r"摘要\s*[:：]\s*([\s\S]+)", response_text)
    if m: summary = m.group(1).strip()
    return {"狀態": status, "摘要": summary}

# ---------- 主程式 ----------
def main():
    st.set_page_config("📄 AI 透明性檢核", layout="wide")
    inject_custom_css()

    st.title("📄 AI 介入透明性 — 九宮格自動檢核")

    with st.sidebar:
        st.header("操作面板")
        uploaded_pdf = st.file_uploader("📥 上傳 PDF 文件", type=["pdf"])
        use_rag = st.checkbox("🔎 啟用向量庫 RAG", value=True)
        analyze_btn = st.button("🚀 開始檢核", use_container_width=True)
        if st.button("🧹 清除結果"):
            for key in ['results', 'df_data']:
                if key in st.session_state: del st.session_state[key]
            st.rerun()

    if uploaded_pdf and analyze_btn:
        with st.spinner("⏳ 正在分析中..."):
            pdf_bytes = uploaded_pdf.read()
            full_text = extract_text_by_line(pdf_bytes)
            prompts = build_transparency_prompts(TRANSPARENCY_PRINCIPLES, full_text, rag_docs_k=3 if use_rag else 0)
            
            results = []
            for i, p in enumerate(TRANSPARENCY_PRINCIPLES):
                resp = get_gemini_response(prompts[i])
                parsed = parse_transparency_response(resp)
                suggestion = gen_missing_suggestion(p) if parsed["摘要"] == "未見相關描述" else ""
                results.append({"id": i+1, "title": p.split('：')[0], "status": parsed["狀態"], "summary": parsed["摘要"], "suggestion": suggestion})
            
            st.session_state['results'] = results
            st.session_state['df_data'] = pd.DataFrame(results)

    # --- 呈現結果 (這裡就是你問的地方) ---
    if 'results' in st.session_state:
        res = st.session_state['results']
        found_count = sum(1 for x in res if x['status'] == "存在")
        st.write(f"### 檢核結果：已符合 {found_count} 項 / 缺失 {9-found_count} 項")

        for row in range(3):
            cols = st.columns(3)
            for col in range(3):
                idx = row * 3 + col
                item = res[idx]
                status_color = "#2ecc71" if item['status'] == "存在" else "#e74c3c"
                
                # --- 核心邏輯插入點 ---
                summary_clean = item['summary'].replace("\n", "<br>")
                suggestion_html = ""
                if item['suggestion']:
                    suggestion_html = f"""
                    <div class="suggestion-box" style="margin-top:auto;">
                        <b>💡 建議補充：</b><br>
                        {item['suggestion'].replace("\n", "<br>")}
                    </div>
                    """

                card_html = f"""
                <div class="flip-card">
                  <div class="flip-card-inner">
                    <div class="flip-card-front">
                      <div style="font-size: 2em; opacity: 0.3; position: absolute; top: 10px; right: 20px;">{item['id']}</div>
                      <div style="font-size: 1.2em; font-weight: bold;">{item['title']}</div>
                      <div class="status-badge" style="background-color: {status_color};">{item['status']}</div>
                      <div style="font-size: 0.7em; margin-top: 20px; opacity: 0.8;">(滑鼠懸停翻轉)</div>
                    </div>
                    <div class="flip-card-back">
                      <div style="font-weight: bold; border-bottom: 2px solid {status_color}; width: 100%; margin-bottom: 10px; padding-bottom: 5px; flex-shrink: 0;">
                        內容摘要
                      </div>
                      <div class="summary-text" style="flex-grow: 1;">{summary_clean}</div>
                      {suggestion_html}
                    </div>
                  </div>
                </div>
                """
                # --- 結束插入點 ---
                
                with cols[col]:
                    st.markdown(card_html, unsafe_allow_html=True)

        st.download_button(label="📥 下載完整 CSV 報告", data=st.session_state['df_data'].to_csv(index=False), file_name="report.csv", mime="text/csv")
    else:
        st.info("請於左側上傳文件並點擊【開始檢核】。")

if __name__ == "__main__":
    main()

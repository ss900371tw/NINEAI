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

# 使用最新穩定版模型
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
    "確保介入開發公平性的過程：說明開發AI系統的過程中，採取哪些具體方法防止或減輕模型在不同群體、用戶或情況下產生偏見 or 不公平結果",
    "外部驗證過程：說明如何將開發完成的AI系統帶到真實或模擬的、與開發環境不同的場所或數據集上進行測試和驗證，以確認其穩定性、準確性和泛化能力",
    "模型表現的量化指標：說明此人工智慧模型的量化評估指標，如模型的準確率、模型的召回率、模型的F1分數、模型的AUC曲線",
    "介入實施和使用的持續維護：說明AI部署和實際使用之後，如何進行後續保養、監控、修復錯誤、處理性能衰退及規劃未來版本更新等工作",
    "更新和持續驗證或公平性評估計劃：說明如何定期對模型重新訓練、功能升級，並透過持續監測和評估系統公平性進行調整，確保其不對特定群體產生偏見以符合臨床需求。"
]

# ---------- 輔助函式 ----------
def inject_custom_css():
    st.markdown("""
    <style>
    .flip-card {
      background-color: transparent;
      width: 100%;
      height: 300px;
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
    .flip-card:hover .flip-card-inner { transform: rotateY(180deg); }
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
      text-align: left;
    }
    .flip-card-back::-webkit-scrollbar { width: 4px; }
    .flip-card-back::-webkit-scrollbar-thumb { background: #cbd5e0; border-radius: 10px; }
    .status-badge { margin-top: 10px; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: bold; color: white; }
    .summary-text { font-size: 0.9em; line-height: 1.5; margin-bottom: 10px; }
    .suggestion-box {
      margin-top: 5px;
      padding: 8px;
      background-color: #fff5f5;
      border-left: 4px solid #f56565;
      font-size: 0.8em;
    }
    </style>
    """, unsafe_allow_html=True)

def clean_text_for_html(text):
    """移除 AI 回傳中可能破壞 HTML 的 Markdown 標籤"""
    if not text: return ""
    # 移除 Markdown 的程式碼區塊標籤
    text = re.sub(r'```[a-z]*', '', text)
    text = text.replace('```', '')
    # 移除過多的星號
    text = text.replace('**', '')
    # 將換行符號轉為 HTML 換行
    text = text.replace('\n', '<br>')
    return text.strip()

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
        return response.text.strip()
    except Exception as e:
        return f"⚠️ 錯誤：{e}"

def gen_missing_suggestion(principle_text):
    prompt = f"你是一位專業 AI 透明性撰寫員。請針對以下缺失提供簡短補強建議（50字內），禁止使用 Markdown 或 HTML：\n\n{principle_text}"
    return get_gemini_response(prompt)

def build_transparency_prompts(principles, full_text, rag_docs_k=3):
    prompts = []
    rag_context = ""
    if vector_store and rag_docs_k > 0:
        docs = vector_store.similarity_search("AI Transparency", k=rag_docs_k)
        rag_context = "\n---\n".join(doc.page_content for doc in docs)

    for p in principles:
        title, desc = p.split('：', 1)
        prompt = f"分析文件是否包含「{title}」。要求：{desc}\n\n文件內容：{full_text[:3500]}\n\n回覆格式：\n狀態: 存在/不存在\n摘要: (內容摘要)"
        prompts.append(prompt)
    return prompts

def parse_transparency_response(response_text):
    status = "存在" if "狀態: 存在" in response_text or "狀態：存在" in response_text else "不存在"
    summary = "未見相關描述"
    m = re.search(r"摘要\s*[:：]\s*([\s\S]+)", response_text)
    if m: summary = m.group(1).strip()
    return {"狀態": status, "摘要": summary}

def main():
    st.set_page_config("📄 AI 透明性檢核", layout="wide")
    inject_custom_css()
    st.title("📄 AI 介入透明性 — 九宮格自動檢核")

    with st.sidebar:
        st.header("操作面板")
        uploaded_pdf = st.file_uploader("📥 上傳 PDF 文件", type=["pdf"])
        use_rag = st.checkbox("🔎 啟用向量庫 RAG", value=True)
        analyze_btn = st.button("🚀 開始檢核", use_container_width=True)

    if uploaded_pdf and analyze_btn:
        with st.spinner("分析中..."):
            pdf_bytes = uploaded_pdf.read()
            full_text = extract_text_by_line(pdf_bytes)
            prompts = build_transparency_prompts(TRANSPARENCY_PRINCIPLES, full_text, 3 if use_rag else 0)
            
            results = []
            for i, p in enumerate(TRANSPARENCY_PRINCIPLES):
                resp = get_gemini_response(prompts[i])
                parsed = parse_transparency_response(resp)
                suggestion = gen_missing_suggestion(p) if parsed["摘要"] == "未見相關描述" else ""
                
                results.append({
                    "id": i+1,
                    "title": p.split('：')[0],
                    "status": parsed["狀態"],
                    "summary": clean_text_for_html(parsed["摘要"]),
                    "suggestion": clean_text_for_html(suggestion)
                })
            st.session_state['results'] = results

    if 'results' in st.session_state:
        res = st.session_state['results']
        for row in range(3):
            cols = st.columns(3)
            for col in range(3):
                idx = row * 3 + col
                item = res[idx]
                status_color = "#2ecc71" if item['status'] == "存在" else "#e74c3c"
                
                # 這裡使用變數預先組合成建議方塊，避免在 f-string 中出現複雜邏輯
                suggestion_html = f'<div class="suggestion-box"><b>💡 建議：</b><br>{item["suggestion"]}</div>' if item['suggestion'] else ""
                
                card_html = f"""
                <div class="flip-card">
                  <div class="flip-card-inner">
                    <div class="flip-card-front">
                      <div style="font-size: 2em; opacity: 0.3; position: absolute; top: 10px; right: 20px;">{item['id']}</div>
                      <div style="font-size: 1.1em; font-weight: bold;">{item['title']}</div>
                      <div class="status-badge" style="background-color: {status_color};">{item['status']}</div>
                    </div>
                    <div class="flip-card-back">
                      <div style="font-weight: bold; border-bottom: 2px solid {status_color}; margin-bottom: 10px;">內容摘要</div>
                      <div class="summary-text">{item['summary']}</div>
                      {suggestion_html}
                    </div>
                  </div>
                </div>
                """
                with cols[col]:
                    st.markdown(card_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

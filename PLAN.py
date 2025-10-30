# app_transparency.py
import os
import re
import tempfile
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from difflib import SequenceMatcher
from PIL import Image
import ollama  # ✅ 改為使用 Ollama
import google.generativeai as genai
from dotenv import load_dotenv
import os

# ---------- 初始化 ----------
# ✅ 初始化 Gemini
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)

# 使用 Gemini 2.5 Pro 模型
model = genai.GenerativeModel("gemini-2.5-pro")
chat = model.start_chat()


# FAISS 向量庫初始化（請確保 INDEX_FILE_PATH 與 embeddings 設定正確）
INDEX_FILE_PATH = "faiss_index"
vector_store = None
try:
    vector_store = FAISS.load_local(INDEX_FILE_PATH, embeddings=HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
except Exception as e:
    vector_store = None
    print("⚠️ 無法載入 FAISS 向量庫：", e)

# ---------- 九大透明性原則定義 ----------
TRANSPARENCY_PRINCIPLES = [
    "介入詳情及輸出：清楚說明AI介入的具體內容及其輸出結果。",
    "介入目的：明確介入的目的及其預期效果。",
    "介入的警告範圍外使用：說明介入在何種情況下可能不適用。",
    "介入開發詳情及輸入特徵：提供開發過程中的詳細資訊及所用的數據特徵。",
    "確保介入開發公平性的過程：確保開發過程中公平性得到保障。",
    "外部驗證過程：進行外部驗證以確保介入的有效性。",
    "模型表現的量化指標：提供量化的指標來評估模型的表現。",
    "介入實施和使用的持續維護：確保介入在實施後持續得到維護。",
    "更新和持續驗證或公平性評估計劃：定期更新和評估介入的公平性。"
]

# ---------- 輔助函式 ----------
def extract_text_by_line(pdf_bytes):
    """使用 PyMuPDF 按 block 取出文字"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    lines = []
    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if text:
                lines.append(text)
    return "\n\n".join(lines)

def get_gemini_response(prompt):
    """使用 Gemini 模型回應"""
    try:
        response = chat.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini 呼叫錯誤：{e}"
        
        
def build_transparency_prompts(principles, full_text, rag_docs_k=3):
    """
    為每一原則建立 prompt。
    """
    prompts = []
    rag_context = ""
    if vector_store:
        merged_query = " ".join(principles[:3])
        try:
            docs = vector_store.similarity_search(merged_query, k=rag_docs_k)
            rag_context = "\n---\n".join(doc.page_content for doc in docs)
        except Exception:
            rag_context = ""

    for p in principles:
        prompt = f"""
你是一位使用繁體中文的審查員，請根據下方「申請文件內容」判斷：
1.是否明確涵蓋下列透明性原則
2.撰寫文件中模型所涉及下列透明性原則的內容（只就文件中明載內容判斷，不得推論或補足）。

---- 要檢核的原則 ----
{p}
---- 文件內容（節錄） ----
{full_text}
---- 向量檢索到的相關參考段落（若有） ----
{rag_context}
---- 回覆格式（請**嚴格**遵守，以利程式解析）----
狀態: 存在 / 不存在
摘要: （一至兩行，說明文件中哪段或如何提及。若不存在，請寫「未發現相關描述」。）
---- 結束 ----
"""
        prompts.append(prompt.strip())
    return prompts

def parse_transparency_response(response_text):
    """解析 Ollama 回應（改良版）"""
    response_text = response_text.strip()
    original = response_text
    status = "無法判讀"
    summary = "未發現相關描述"

    m = re.search(r"狀態\s*[:：]\s*(存在|不存在)", response_text)
    if m:
        status = m.group(1).strip()
    else:
        if "存在" in response_text and "不存在" not in response_text:
            status = "存在"
        elif "不存在" in response_text:
            status = "不存在"

    m2 = re.search(r"摘要\s*[:：]\s*(.+?)(?:\n|$)", response_text, flags=re.DOTALL)
    if m2:
        summary = m2.group(1).strip()
    else:
        summary = original[:300].replace("\n", " ").strip()

    return {"狀態": status, "摘要": summary}

# ---------- 主流程與 UI ----------
def main():
    st.set_page_config("📄 AI 介入透明性檢核", layout="wide")
    st.title("📄 單一 PDF — 九大透明性原則自動檢核 (Ollama)")
    st.markdown("上傳單一 PDF，系統會逐條檢查九大透明性原則是否在文件中明載，並產生可下載的 CSV 檔。")

    uploaded_pdf = st.file_uploader("📥 上傳 PDF 文件（單一檔案）", type=["pdf"], accept_multiple_files=False)
    use_rag = st.checkbox("🔎 啟用向量庫（若已載入 FAISS，可使用 RAG 上下文）", value=True)
    analyze_btn = st.button("🚀 開始檢核")

    if uploaded_pdf and analyze_btn:
        pdf_bytes = uploaded_pdf.read()
        pdf_filename = uploaded_pdf.name.rsplit(".", 1)[0]

        with st.spinner("⏳ 讀取 PDF 並分析中，請稍候..."):
            full_text = extract_text_by_line(pdf_bytes)
            prompts = build_transparency_prompts(
                TRANSPARENCY_PRINCIPLES, full_text,
                rag_docs_k=3 if use_rag and vector_store else 0
            )

            results = []
            for i, p in enumerate(TRANSPARENCY_PRINCIPLES):
                prompt = prompts[i]
                resp = get_gemini_response(prompt)
                parsed = parse_transparency_response(resp)
                results.append({
                    "原則編號": i+1,
                    "原則名稱": p,
                    "狀態": parsed["狀態"],
                    "摘要": parsed["摘要"],
                })

        df = pd.DataFrame(results)
        df = df[["原則編號", "原則名稱", "狀態", "摘要"]]

        st.success("✅ 檢核完成")
        st.markdown(f"檔案：**{uploaded_pdf.name}**  → 共有 {len(df)} 項檢核結果")
        st.dataframe(df, use_container_width=True)

        csv_data = df.to_csv(index=False)
        filename = f"{pdf_filename}_九大透明性檢核.csv"
        st.download_button(
            label=f"📥 下載 CSV：{filename}",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )

        for idx, row in df.iterrows():
            with st.expander(f"🔎 第 {row['原則編號']} 項：{row['原則名稱']} — 狀態：{row['狀態']}"):
                st.markdown(f"**摘要**：{row['摘要']}")


    elif not uploaded_pdf:
        st.info("請先上傳一份 PDF，然後按【開始檢核】。")

if __name__ == "__main__":
    main()

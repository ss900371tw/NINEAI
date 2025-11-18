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
    "介入詳情及輸出：說明人工智慧模型的基本特徵（如模型架構、訓練技術）以及模型輸出的形式。",
    "介入目的：說明人工智慧模型設計的核心目標以及適用情境。",
    "介入的警告範圍外使用：說明人工智慧模型適用、不適用範圍，及其可能發生之風險。",
    "介入開發詳情及輸入特徵：說明人工智慧模型核心技術，包含數據集、模型結構、訓練方法等。",
    "確保介入開發公平性的過程：說明人工智慧模型開發過程，數據集平衡方式。",
    "外部驗證過程：說明外部驗證與評估過程。",
    "模型表現的量化指標：說明此人工智慧模型的量化評估指標，如模型的準確率、模型的召回率、模型的F1分數、模型的AUC曲線",
    "介入實施和使用的持續維護：說明模型部署後如何進行持續維護，包括性能監控、錯誤修復及更新。",
    "更新和持續驗證或公平性評估計劃：說明如何定期重新訓練模型、更新數據集，並進行持續性驗證與公平性評估，讓模型效能穩定且符合公平性標準，以符合臨床需求。"
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
    """使用 Gemini 模型回應（改為無狀態的 generate_content）"""
    try:
        # 錯誤的用法： response = chat.send_message(prompt)
        # 正確的用法：
        response = model.generate_content(prompt) 
        
        return response.text.strip()
    except Exception as e:
        # 檢查是否有因為內容過長或安全設定而被阻擋
        try:
            # 嘗試讀取更詳細的錯誤（如果 API 回應中有）
            error_details = str(e)
            if response.prompt_feedback:
                 error_details = f"Prompt blocked: {response.prompt_feedback}"
            elif response.candidates and response.candidates[0].finish_reason != 'STOP':
                 error_details = f"Generation stopped: {response.candidates[0].finish_reason}"
            return f"⚠️ Gemini 呼叫錯誤：{error_details}"
        except:
             return f"⚠️ Gemini 呼叫錯誤：{e}"


def gen_missing_suggestion(principle_text):
    """若文件未涵蓋某透明性原則，請 Gemini 生成建議補充內容"""
    prompt = f"""
你是一位專業 AI 模型透明性報告撰寫員。
下列為透明性原則說明，請寫出「若要補上本原則，你會怎麼撰寫？」以符合標準。

透明性原則內容：
{principle_text}

請用繁體中文撰寫，語氣正式、能直接貼入報告文件。
"""
    resp = get_gemini_response(prompt)
    return resp.strip()
        
        
def build_transparency_prompts(principles, full_text, rag_docs_k=3):
    """
    為每一原則建立 prompt。
    """
    prompts = []
    rag_context = ""
    if vector_store:
        merged_query = " ".join(principles)
        try:
            docs = vector_store.similarity_search(merged_query, k=rag_docs_k)
            rag_context = "\n---\n".join(doc.page_content for doc in docs)
        except Exception:
            rag_context = ""

    for p in principles:
        prompt = f"""
---- 要請你說明的透明性原則 ----
{p.split('：', 1)[0]}
你是一位使用繁體中文的透明性原則講解員，請根據下方「申請文件內容」判斷：
1.是否存在相關描述讓你可以 {p.split('：', 1)[1]}
2.請 {p.split('：', 1)[1]}
---- 文件內容（節錄） ----
{full_text}
---- 向量檢索到的相關參考段落（若有） ----
{rag_context}
---- 回覆格式（請**嚴格**遵守，以利程式解析）----
狀態:  存在 / 不存在
摘要: （{p.split('：', 1)[1]}。若不存在，請寫「未發現相關描述」。）

----注意----
請勿直接複製文件中的符號或段落，請自行用通順中文摘要說明。
"""
        prompts.append(prompt.strip())
    return prompts

def parse_transparency_response(response_text):
    response_text = response_text.strip()
    original = response_text
    status = "無法判讀"
    summary = "未發現相關描述"

    # --- 判斷狀態 ---
    m = re.search(r"狀態\s*[:：]\s*(存在|不存在)", response_text)
    if m:
        status = m.group(1).strip()
    else:
        if "存在" in response_text and "不存在" not in response_text:
            status = "存在"
        elif "不存在" in response_text:
            status = "不存在"

    # --- 抓取摘要內容 ---
    m2 = re.search(r"摘要\s*[:：]\s*([\s\S]+)", response_text)
    if m2:
        summary = m2.group(1).strip()
    else:
        summary = original.replace("\n", " ").strip()

    # ✅ 強制規則：若狀態為「不存在」，摘要改為「未見相關描述」
    if status == "不存在":
        summary = "未見相關描述"

    return {"狀態": status, "摘要": summary}

# ---------- 主流程與 UI ----------
def main():
    st.set_page_config("📄 AI 介入透明性檢核", layout="wide")
    st.title("📄 單一 PDF — 九大透明性原則自動檢核 (Gemini)")
    st.markdown("上傳單一 PDF，系統會逐條檢查九大透明性原則是否在文件中明載，並產生可下載的 CSV 檔。")

    uploaded_pdf = st.file_uploader("📥 上傳 IRB WORD 或 PDF 文件（單一檔案）", type=["pdf","docx"], accept_multiple_files=False)
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

                suggestion = ""
                if parsed["摘要"] == "未見相關描述":
                    suggestion = gen_missing_suggestion(p)

                results.append({
                    "原則編號": i+1,
                    "原則名稱": p,
                    "狀態": parsed["狀態"],
                    "摘要": parsed["摘要"],
                    "建議補充內容": suggestion,
                })

        df = pd.DataFrame(results)
        df = df[["原則編號", "原則名稱", "狀態", "摘要", "建議補充內容"]]

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
                if row["摘要"] == "未見相關描述":
                    st.markdown(f"**建議補充內容**：{row['建議補充內容']}")

    elif not uploaded_pdf:
        st.info("請先上傳一份 PDF，然後按【開始檢核】。")

if __name__ == "__main__":
    main()

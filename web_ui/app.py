import streamlit as st
import os

# 全局显卡可见性 (Retrieval: 0, LLM: 1,2)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

from core.retrieval import RetrievalSystem
from core.llm_engine import VLLMEngine

st.set_page_config(layout="wide", page_title="DualSight-RAG v2", page_icon="👁️")

# --- CSS ---
st.markdown("""
<style>
    .score-tag {
        font-size: 0.8em; color: #555; background: #eee; 
        padding: 2px 5px; border-radius: 4px; margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 系统加载 ---
@st.cache_resource
def load_system():
    # 这里的路径需对应 ingest 生成的路径
    retriever = RetrievalSystem(
        index_root_path="vectorstore/energy",
        doc_store_path="vectorstore/energy/doc_store.json"
    )
    llm = VLLMEngine()
    return retriever, llm

if "system_loaded" not in st.session_state:
    with st.spinner("🚀 Booting SigLIP Fragment-Aggregation System..."):
        try:
            retriever, llm = load_system()
            st.session_state.retriever = retriever
            st.session_state.llm = llm
            st.session_state.system_loaded = True
            st.success("✅ System Online")
        except Exception as e:
            st.error(f"Init Failed: {e}")
            st.stop()
else:
    retriever = st.session_state.retriever
    llm = st.session_state.llm

# --- UI 逻辑 ---
st.title("👁️ DualSight-RAG (Fragment Aggregation)")

if prompt := st.chat_input("Ask about charts or documents..."):
    st.chat_message("user").write(prompt)

    # 1. 检索与融合
    with st.status("🔍 Retrieving & Aggregating Fragments...", expanded=True) as status:
        # 这一步内部完成了：碎片召回 -> Doc聚合 -> 公式打分 -> 双路融合
        results = retriever.search_and_rerank(prompt, top_k_final=5)
        
        evidence_text = ""
        if results:
            status.write(f"✅ Found {len(results)} docs after fusion.")
            cols = st.columns(3)
            for i, res in enumerate(results):
                # 获取分项得分
                s_final = res['final_score']
                s_txt = res['scores']['text']
                s_vis = res['scores']['visual']
                
                # 构造 Prompt 上下文
                evidence_text += f"\n[Doc {i+1}]: {res['content']}\n"
                
                # UI 展示
                if i < 3:
                    with cols[i]:
                        st.image(res['path'])
                        st.markdown(f"**Rank {i+1}** (Score: {s_final:.3f})")
                        st.caption(f"Text: {s_txt:.3f} | Vis: {s_vis:.3f}")
                        with st.expander("Show Text"):
                            st.text(res['content'][:200] + "...")
        else:
            status.write("⚠️ No relevant documents found.")
            
        status.update(label="Retrieval Complete", state="complete")

    # 2. 生成回答
    if evidence_text:
        with st.chat_message("assistant"):
            with st.spinner("🧠 Reasoning..."):
                final_prompt = f"""
                Context:
                {evidence_text}
                
                User Question: "{prompt}"
                
                Answer the question based on the Context.
                """
                response = llm.generate(final_prompt)
                st.write(response)
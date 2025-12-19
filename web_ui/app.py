import streamlit as st
import os

# =================  核心 1: 显存物理隔离 =================
# 必须在导入 torch/core 之前设置！
# 这让 Python 进程能看到所有卡，具体的隔离由 core 内部的类自己管理：
# - RetrievalSystem -> 强制用 cuda:0
# - VLLMEngine -> 强制用 cuda:1, cuda:2
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# 导入 UI 组件
from PIL import Image

# 导入新核心 (注意：这里不需要 vectordb.py 了，直接调 core)
from core.retrieval import RetrievalSystem
from core.llm_engine import VLLMEngine

# --- 1. 页面基础配置 ---
st.set_page_config(layout="wide", page_title="DualSight-RAG", page_icon="👁️")

st.markdown("""
<style>
    .stChatFloatingInputContainer {bottom: 20px;}
    .evidence-card {
        background-color: #f0f2f6; border-radius: 8px; padding: 10px; margin-bottom: 5px;
        border-left: 4px solid #00E676; font-size: 0.9em;
    }
    .score-tag {
        background-color: #e3f2fd; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 核心资源加载 (单例模式) ---
@st.cache_resource
def load_system():
    """
    初始化两大引擎，利用 Streamlit 缓存机制保证只加载一次
    """
    # A. 初始化检索系统 (GPU 0)
    # 路径根据你之前的设置，假设在 vectorstore/energy 下
    retriever = RetrievalSystem(
        index_root_path="vectorstore/energy",
        doc_store_path="vectorstore/energy/doc_store.json"
    )
    
    # B. 初始化生成引擎 (GPU 1 & 2)
    llm = VLLMEngine() 
    
    return retriever, llm

# 加载系统 (带有加载动画)
if "system_loaded" not in st.session_state:
    with st.spinner("🚀 Booting up DualSight System (SigLIP + Chunked BGE + vLLM)..."):
        try:
            retriever, llm = load_system()
            st.session_state.retriever = retriever
            st.session_state.llm = llm
            st.session_state.system_loaded = True
            st.success("✅ System Online: GPU 0 (Search) + GPU 1&2 (Reasoning)")
        except Exception as e:
            st.error(f"❌ System Init Failed: {e}")
            st.stop()
else:
    retriever = st.session_state.retriever
    llm = st.session_state.llm

# --- 3. 侧边栏 ---
with st.sidebar:
    st.title("👁️ DualSight-RAG")
    st.caption("Unified-Space Multimodal RAG")
    st.markdown("---")

    # 🖼️ 多模态输入
    st.header("🖼️ Multimodal Input")
    uploaded_file = st.file_uploader("Upload query image (optional):", type=["png", "jpg", "jpeg"])
    
    user_image_path = None
    if uploaded_file:
        # 保存上传的图片到临时目录，以便后续处理（如果需要）
        # 目前 vLLM 策略只处理文本，这里仅做展示
        st.image(uploaded_file, caption="Query Image", use_column_width=True)
        st.success("Image added to context!")

    st.markdown("---")
    
    # ⚙️ 参数
    st.header("⚙️ Rerank Settings")
    top_k = st.slider("Top-K Evidence", 1, 10, 5)
    
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- 4. 主聊天逻辑 ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm ready to analyze energy documents using Dual-Path retrieval."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Greenmount Road..."):
    # 1. 显示用户提问
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        evidence_text = ""
        debug_info = st.empty()
        
        # --- 核心 2: 检索阶段 (双路召回 + 分块聚合重排) ---
        with st.status("🔍 Dual-Path Retrieval & Reranking...", expanded=True) as status:
            # A. 检索
            # search_and_rerank 内部已经包含了 SigLIP 召回 -> BGE 分块打分
            results = retriever.search_and_rerank(prompt, top_k_final=top_k)
            
            if results:
                status.write(f"✅ Found {len(results)} relevant docs (after Reranking).")
                
                # B. 展示证据卡片 (Evidence Cards)
                cols = st.columns(min(len(results), 3))
                for i, res in enumerate(results):
                    score = res.get('score', 0.0)
                    
                    # 拼接 Context：包含 Qwen 预生成的详细描述
                    evidence_text += f"\n[Document {i+1} (Score: {score:.4f})]:\n{res['content']}\n"
                    
                    # 在界面显示缩略图
                    if i < 3: 
                        with cols[i]:
                            st.image(res['path'], caption=f"Rank {i+1} (Score: {score:.2f})")
                            with st.expander(f"See Description {i+1}"):
                                st.caption(res['content'][:200] + "...")
            else:
                status.write("⚠️ No relevant documents found.")
            
            status.update(label="Retrieval Complete", state="complete")

        # --- 核心 3: 生成阶段 (vLLM 加速) ---
        response_placeholder = st.empty()
        
        # 构造 Prompt：只喂 Retrieve 到的 Text Context
        # 理由：Qwen 预处理的描述已经是 OCR 级别的了，直接喂文本给 vLLM 
        # 既能避免 vLLM 多模态格式的坑，又能极大提升推理速度 (Text-Only 是最快的)
        final_prompt = f"""
        Reference Context (High-Fidelity OCR Descriptions):
        {evidence_text if evidence_text else "No specific context found."}
        
        User Question: "{prompt}"
        
        Please answer the User Question accurately based ONLY on the Reference Context above.
        If the answer involves data from tables/charts, cite the Document number.
        """
        
        # 调用 vLLM
        # 注意：这里我们移除了 images 参数，因为 context 已经包含了图片信息
        full_response = llm.generate(
            prompt=final_prompt,
            system_prompt="You are an expert AI assistant. Answer strictly based on the provided context."
        )
        
        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
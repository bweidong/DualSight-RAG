import os
import json
import torch
import faiss
from transformers import AutoModel, AutoProcessor
# 导入刚才写的 Reranker
from core.reranker import BGEChunkedReranker

# 路径配置 (根据你的实际情况修改)
SIGLIP_PATH = ""
BGE_PATH = ""

class RetrievalSystem:
    def __init__(self, index_root_path, doc_store_path):
        self.device = "cuda:0" # 严格限制在 0 号卡
        print(f"Initializing Retrieval System on {self.device}...")

        # 1. SigLIP (检索路)
        self.siglip_model = AutoModel.from_pretrained(SIGLIP_PATH, local_files_only=True).to(self.device).eval()
        self.siglip_processor = AutoProcessor.from_pretrained(SIGLIP_PATH, local_files_only=True)

        # 2. BGE (重排路)
        self.reranker = BGEChunkedReranker(BGE_PATH, device=self.device)

        # 3. FAISS (CPU 加载)
        print("   - Loading FAISS Indices...")
        self.index_text = faiss.read_index(os.path.join(index_root_path, "faiss_text.bin"))
        self.index_image = faiss.read_index(os.path.join(index_root_path, "faiss_image.bin"))
        
        # 4. 文档库
        with open(doc_store_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.doc_store = {int(k): v for k, v in data.items()}

    def _get_vec(self, text):
        with torch.no_grad():
            inputs = self.siglip_processor(text=[text], return_tensors="pt", padding="max_length", truncation=True).to(self.device)
            emb = self.siglip_model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            return emb.cpu().numpy()

    def search_and_rerank(self, query, top_k_final=5):
        # 1. 粗排 (Recall) - 取 Top 100
        q_vec = self._get_vec(query)
        candidates_map = {} 
        
        # 文本路 (0.6) + 视觉路 (0.4)
        D_t, I_t = self.index_text.search(q_vec, 100)
        D_i, I_i = self.index_image.search(q_vec, 100)
        
        for s, i in zip(D_t[0], I_t[0]): 
            if i != -1: candidates_map[i] = candidates_map.get(i, 0) + s * 0.6
        for s, i in zip(D_i[0], I_i[0]): 
            if i != -1: candidates_map[i] = candidates_map.get(i, 0) + s * 0.4
            
        # 2. 准备精排数据
        candidates = []
        for idx in candidates_map:
            if idx in self.doc_store:
                candidates.append({
                    "id": idx,
                    "content": self.doc_store[idx]['content'],
                    "path": self.doc_store[idx]['path'],
                    "initial_score": candidates_map[idx]
                })
        
        # 3. 分块聚合精排
        return self.reranker.rerank(query, candidates, top_k=top_k_final)
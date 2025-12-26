import os
import json
import torch
import faiss
import numpy as np
from transformers import AutoModel, AutoProcessor

# 路径配置
SIGLIP_PATH = "/data/models/google/siglip-so400m-patch14-384"

class RetrievalSystem:
    def __init__(self, index_root_path, doc_store_path):
        self.device = "cuda:0"
        print(f"Initializing Fragment-Level Retrieval on {self.device}...")

        # 1. 模型 (仅用于 Query 编码)
        self.siglip_model = AutoModel.from_pretrained(SIGLIP_PATH, local_files_only=True).to(self.device).eval()
        self.siglip_processor = AutoProcessor.from_pretrained(SIGLIP_PATH, local_files_only=True)

        # 2. 加载索引
        print("   - Loading FAISS Indices...")
        # 注意加载的是 chunks 索引
        self.index_text = faiss.read_index(os.path.join(index_root_path, "faiss_text_chunks.bin"))
        self.index_image = faiss.read_index(os.path.join(index_root_path, "faiss_image.bin"))
        
        # 3. 加载映射表
        with open(doc_store_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.doc_store = {int(k): v for k, v in data.items()}
            
        chunk_map_path = os.path.join(index_root_path, "chunk_map.json")
        with open(chunk_map_path, 'r', encoding='utf-8') as f:
            # key 是 string, 需要转 int
            self.chunk_to_doc = {int(k): int(v) for k, v in json.load(f).items()}

    def _get_query_vec(self, text):
        with torch.no_grad():
            inputs = self.siglip_processor(text=[text], return_tensors="pt", padding="max_length", truncation=True).to(self.device)
            emb = self.siglip_model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            return emb.cpu().numpy()

    def _aggregate_scores(self, scores_list):
        """
        聚合公式: Score = alpha * Max + (1 - alpha) * Mean_TopK
        """
        if not scores_list: return 0.0
        
        # 参数配置
        ALPHA = 0.6
        K = 3
        
        scores = np.array(scores_list)
        
        # 1. Max Score
        max_score = np.max(scores)
        
        # 2. Top-K Mean
        # 如果碎片不够 K 个，就取所有的平均
        k_real = min(K, len(scores))
        # partition 找出最大的 k 个 (未排序), 再取 mean
        if k_real > 0:
            top_k_vals = scores[np.argpartition(scores, -k_real)[-k_real:]]
            mean_score = np.mean(top_k_vals)
        else:
            mean_score = 0.0
            
        final_score = ALPHA * max_score + (1 - ALPHA) * mean_score
        return final_score

    def search_and_rerank(self, query, top_k_final=5):
        # 1. Query 编码
        q_vec = self._get_query_vec(query)

        # ================= A. 文本路 (Fragment Search -> Aggregation) =================
        # 1. 检索 Top 100 碎片
        D_t, I_t = self.index_text.search(q_vec, 100)
        
        # 2. GROUP BY Doc_ID
        doc_fragment_scores = {} # {doc_id: [score1, score2...]}
        
        for score, chunk_idx in zip(D_t[0], I_t[0]):
            if chunk_idx == -1: continue
            
            doc_id = self.chunk_to_doc.get(chunk_idx)
            if doc_id is not None:
                if doc_id not in doc_fragment_scores:
                    doc_fragment_scores[doc_id] = []
                doc_fragment_scores[doc_id].append(score)
        
        # 3. 应用聚合公式
        text_candidates = {} # {doc_id: agg_score}
        for doc_id, scores in doc_fragment_scores.items():
            agg_score = self._aggregate_scores(scores)
            text_candidates[doc_id] = agg_score

        # ================= B. 视觉路 (Direct Search) =================
        # 检索 Top 50 图片 (Doc Level)
        D_i, I_i = self.index_image.search(q_vec, 50)
        
        visual_candidates = {} # {doc_id: score}
        for score, doc_id in zip(D_i[0], I_i[0]):
            if doc_id != -1:
                visual_candidates[doc_id] = score

        # ================= C. 双路融合 (Fusion) =================
        # 公式: Final = w1 * Visual + w2 * Text
        W_VISUAL = 0.4
        W_TEXT = 0.6
        
        final_results_map = {}
        all_doc_ids = set(text_candidates.keys()) | set(visual_candidates.keys())
        
        for doc_id in all_doc_ids:
            if doc_id not in self.doc_store: continue
            
            s_text = text_candidates.get(doc_id, 0.0)
            s_visual = visual_candidates.get(doc_id, 0.0)
            
            final_score = (W_VISUAL * s_visual) + (W_TEXT * s_text)
            
            final_results_map[doc_id] = {
                "id": doc_id,
                "content": self.doc_store[doc_id]['content'],
                "path": self.doc_store[doc_id]['path'],
                "final_score": final_score,
                "scores": {"text": s_text, "visual": s_visual}
            }
            
        # 排序并截断
        results = list(final_results_map.values())
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results[:top_k_final]
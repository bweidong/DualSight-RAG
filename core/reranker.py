import torch
import gc
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BGEChunkedReranker:
    def __init__(self, model_path, device="cuda:0"):
        print(f"Loading BGE-Reranker (Chunked Mode) on {device}...")
        self.device = device
        self.max_length = 512
        self.chunk_size = 384  # 留窗口给 Query
        self.overlap = 64
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16 # 强制半精度，省显存
        ).to(self.device).eval()

    def compute_score(self, query, doc_text):
        if not doc_text: return -10.0
        
        # 1. 切分长文本 (Sliding Window)
        tokens = self.tokenizer.tokenize(doc_text)
        chunks = []
        if len(tokens) <= self.chunk_size:
            chunks = [doc_text]
        else:
            stride = self.chunk_size - self.overlap
            for i in range(0, len(tokens), stride):
                chunk_tokens = tokens[i : i + self.chunk_size]
                chunks.append(self.tokenizer.convert_tokens_to_string(chunk_tokens))
                if i + self.chunk_size >= len(tokens): break
        
        # 2. 构造 Pair
        pairs = [[query, c] for c in chunks]
        
        # 3. 批量打分
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True, max_length=512, return_tensors='pt'
            ).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            
            # 聚合策略: Max Pooling (取最相关的一段作为文档分)
            final_score = torch.max(scores).item()
            
        return final_score

    def rerank(self, query, candidates, top_k=5):
        """
        candidates: list of dict, must contain 'content'
        """
        if not candidates: return []
        
        scored_results = []
        for doc in candidates:
            # 这里的 content 是 Qwen 生成的长描述
            s = self.compute_score(query, doc['content'])
            doc['score'] = s
            scored_results.append(doc)
            
        # 降序排列
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return scored_results[:top_k]
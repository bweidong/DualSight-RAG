import os
import sys
import json
import torch
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

# --- 路径适配 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModel, AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

# 配置部分 (可提取到 configs/config.py)
MODEL_PATHS = {
    "qwen": "/data/models/Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    "siglip": "/data/models/google/siglip-so400m-patch14-384"
}
VECTOR_STORE_ROOT = "vectorstore"

# ================= 模型初始化 =================
print("🔄 Loading Models for Ingestion...")

# 1. Qwen (用于生成描述)
try:
    caption_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATHS['qwen'],
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
        max_memory={0: "1GB", 1: "22GB", 2: "22GB"}, # 预留 0 号卡给 SigLIP
        local_files_only=True,
        trust_remote_code=True
    )
    caption_processor = AutoProcessor.from_pretrained(MODEL_PATHS['qwen'], local_files_only=True, trust_remote_code=True)
except Exception as e:
    print(f"❌ Qwen Load Failed: {e}")
    sys.exit(1)

# 2. SigLIP (用于生成向量)
siglip_model = AutoModel.from_pretrained(MODEL_PATHS['siglip'], local_files_only=True).to("cuda:0").eval()
siglip_processor = AutoProcessor.from_pretrained(MODEL_PATHS['siglip'], local_files_only=True)

print("✅ Models Loaded.")

# ================= 核心工具函数 =================

def generate_description(image_path):
    """Qwen 生成详细描述"""
    prompt = "Describe this image in detail. Extract all text, data from tables, and key visual features."
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    text = caption_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = caption_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(caption_model.device)

    with torch.no_grad():
        generated_ids = caption_model.generate(**inputs, max_new_tokens=1024)
    
    output_text = caption_processor.batch_decode(
        [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True
    )[0]
    return output_text

def chunk_text(text, chunk_size=256, overlap=32):
    """文本切片: 简单的滑动窗口"""
    # 简单按空格分词，中文环境建议改用 list(text) 或结巴分词
    words = text.split() 
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_siglip_embeddings(text_chunks=None, image_path=None):
    """SigLIP 统一编码"""
    txt_embs = None
    img_emb = None
    
    with torch.no_grad():
        # 1. 文本批量编码
        if text_chunks:
            inputs = siglip_processor(
                text=text_chunks, return_tensors="pt", padding="max_length", truncation=True, max_length=64
            ).to("cuda:0")
            e = siglip_model.get_text_features(**inputs)
            e = e / e.norm(p=2, dim=-1, keepdim=True)
            txt_embs = e.cpu().numpy()

        # 2. 图片编码
        if image_path:
            img_obj = Image.open(image_path).convert("RGB")
            inputs = siglip_processor(images=img_obj, return_tensors="pt").to("cuda:0")
            e = siglip_model.get_image_features(**inputs)
            e = e / e.norm(p=2, dim=-1, keepdim=True)
            img_emb = e.cpu().numpy()
            
    return txt_embs, img_emb

# ================= 主流程 =================

def build_database(dataset_name):
    raw_data_dir = os.path.join("raw_data", dataset_name)
    output_dir = os.path.join(VECTOR_STORE_ROOT, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    images = [f for f in os.listdir(raw_data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"🚀 Processing {len(images)} images from [{dataset_name}]...")

    # 索引定义
    dim = 1152 # SigLIP Large embedding size
    index_text = faiss.IndexFlatIP(dim)  # 存储碎片向量
    index_image = faiss.IndexFlatIP(dim) # 存储整图向量

    doc_store = {}        # DocID -> {content, path}
    chunk_map = {}        # ChunkID -> DocID
    global_chunk_id = 0   # 碎片全局计数器

    for doc_id, img_file in tqdm(enumerate(images), total=len(images)):
        img_path = os.path.join(raw_data_dir, img_file)
        
        try:
            # 1. 生成描述
            desc = generate_description(img_path)
            
            # 2. 文本切片
            chunks = chunk_text(desc)
            
            # 3. 编码 (文本碎片 + 图片)
            txt_vecs, img_vec = get_siglip_embeddings(text_chunks=chunks, image_path=img_path)
            
            # 4. 存入索引
            # A. 文本碎片入库
            if txt_vecs is not None:
                index_text.add(txt_vecs)
                # 记录映射关系
                for _ in range(len(chunks)):
                    chunk_map[global_chunk_id] = doc_id
                    global_chunk_id += 1
            
            # B. 图片入库
            if img_vec is not None:
                index_image.add(img_vec) # Visual Index 下标直接对应 DocID
            
            # 5. 存元数据
            doc_store[doc_id] = {
                "path": img_path,
                "content": desc, # 存储完整描述供 LLM 阅读
                "filename": img_file
            }

        except Exception as e:
            print(f"⚠️ Error on {img_file}: {e}")

    # 保存文件
    print("💾 Saving indices and maps...")
    faiss.write_index(index_text, os.path.join(output_dir, "faiss_text_chunks.bin")) # 注意文件名变化
    faiss.write_index(index_image, os.path.join(output_dir, "faiss_image.bin"))
    
    with open(os.path.join(output_dir, "doc_store.json"), "w", encoding='utf-8') as f:
        json.dump(doc_store, f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(output_dir, "chunk_map.json"), "w", encoding='utf-8') as f:
        json.dump(chunk_map, f)

    print(f"🎉 Database built! Total Chunks: {global_chunk_id}, Total Docs: {len(doc_store)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    build_database(args.dataset)
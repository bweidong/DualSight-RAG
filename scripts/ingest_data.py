import os
import sys
import json
import torch
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

# --- 0. 路径适配 (为了能导入 configs) ---
# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModel, AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info


import transformers.activations
if not hasattr(transformers.activations, "PytorchGELUTanh"):
    transformers.activations.PytorchGELUTanh = transformers.activations.NewGELUActivation
# ====================================================================

# 🟢 从配置文件导入路径 (避免硬编码)
try:
    from configs.config import MODEL_PATHS, VECTOR_STORE_ROOT
except ImportError:
    # 如果用户没配好环境，提供默认值或报错
    print("⚠️ Warning: Could not import configs. Using hardcoded paths.")
    MODEL_PATHS = {
        "qwen": "/data/models/Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        "siglip": "/data/models/google/siglip-so400m-patch14-384"
    }
    VECTOR_STORE_ROOT = "vectorstore"

# ================= 1. 初始化模型 =================
print("🔄 Loading Models for Ingestion...")

# --- A. 加载 Qwen (Transformers 模式) ---
print(f"   - Loading Qwen2.5-VL from {MODEL_PATHS['qwen']}...")
try:
    caption_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATHS['qwen'],
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
        max_memory={0: "1GB", 1: "22GB", 2: "22GB"}, # 预留 GPU0 给 SigLIP
        local_files_only=True,
        trust_remote_code=True
    )
    caption_processor = AutoProcessor.from_pretrained(MODEL_PATHS['qwen'], local_files_only=True, trust_remote_code=True)
except Exception as e:
    print(f"❌ Failed to load Qwen: {e}")
    sys.exit(1)

# --- B. 加载 SigLIP ---
print(f"   - Loading SigLIP from {MODEL_PATHS['siglip']}...")
siglip_model = AutoModel.from_pretrained(MODEL_PATHS['siglip'], local_files_only=True).to("cuda:0")
siglip_processor = AutoProcessor.from_pretrained(MODEL_PATHS['siglip'], local_files_only=True)

print("✅ All Models Loaded.")

# ================= 2. 核心功能 =================

def generate_description(image_path):
    """Qwen 生成精炼描述"""
    prompt = "Detailedly describe this image. If it contains a document, table, or graph, transcribe the text, headers, and data values verbatim."
    
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    # 构造输入
    text = caption_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = caption_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(caption_model.device)

    # 生成
    with torch.no_grad():
        generated_ids = caption_model.generate(**inputs, max_new_tokens=512)
    
    # 解码
    output_text = caption_processor.batch_decode(
        [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True
    )[0]
    
    return output_text

def get_dual_embeddings(image_path, text_description):
    """SigLIP 生成双路向量"""
    with torch.no_grad():
        # Text Path
        txt_inputs = siglip_processor(
            text=[text_description], return_tensors="pt", padding="max_length", truncation=True
        ).to("cuda:0")
        txt_emb = siglip_model.get_text_features(**txt_inputs)
        txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
        
        # Image Path
        img_obj = Image.open(image_path).convert("RGB")
        img_inputs = siglip_processor(images=img_obj, return_tensors="pt").to("cuda:0")
        img_emb = siglip_model.get_image_features(**img_inputs)
        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)

    return txt_emb.cpu().numpy(), img_emb.cpu().numpy()

# ================= 3. 主流程 =================

def build_database(dataset_name):
    # 使用配置中的 ROOT 路径
    raw_data_dir = os.path.join("raw_data", dataset_name)
    output_dir = os.path.join(VECTOR_STORE_ROOT, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(raw_data_dir):
        print(f"❌ Data folder not found: {raw_data_dir}")
        return

    images = [f for f in os.listdir(raw_data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"🚀 Processing {len(images)} images from [{dataset_name}]...")

    index_text = faiss.IndexFlatIP(1152)
    index_image = faiss.IndexFlatIP(1152)
    doc_store = {}

    for idx, img_file in tqdm(enumerate(images), total=len(images)):
        img_path = os.path.join(raw_data_dir, img_file)
        try:
            # 1. Captioning
            desc = generate_description(img_path)
            
            # 2. Embedding
            v_txt, v_img = get_dual_embeddings(img_path, desc)
            
            # 3. Indexing
            index_text.add(v_txt)
            index_image.add(v_img)
            
            # 4. Storage
            doc_store[idx] = {
                "path": img_path, # 这里的路径在不同机器上可能需要调整，生产环境建议存相对路径
                "content": desc,
                "original_filename": img_file
            }
            
            if idx == 0: print(f"\n✨ Preview: {desc[:100]}...\n")

        except Exception as e:
            print(f"⚠️ Error on {img_file}: {e}")

    # 保存
    faiss.write_index(index_text, os.path.join(output_dir, "faiss_text.bin"))
    faiss.write_index(index_image, os.path.join(output_dir, "faiss_image.bin"))
    with open(os.path.join(output_dir, "doc_store.json"), "w", encoding='utf-8') as f:
        json.dump(doc_store, f, indent=4, ensure_ascii=False)

    print(f"🎉 Success! Vectors saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset folder name")
    args = parser.parse_args()
    build_database(args.dataset)
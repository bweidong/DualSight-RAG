import json
import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from core.retrieval import RetrievalSystem

def run_evaluation(dataset_name):
    # --- 路径配置 ---
    gt_path = f"benchmark_gt_{dataset_name}.json"
    
    # 指向 vectorstore 根目录
    base_dir = os.path.join("vectorstore", dataset_name)
    doc_path = os.path.join(base_dir, "doc_store.json")

    if not os.path.exists(gt_path):
        print(f"❌ Ground truth not found: {gt_path}")
        return

    print(f"🚀 Initializing Dual-Path Retrieval System for [{dataset_name}]...")
    
    # 初始化检索器
    retriever = RetrievalSystem(index_root_path=base_dir, doc_store_path=doc_path)

    # --- 加载数据 ---
    with open(gt_path, "r", encoding='utf-8') as f:
        raw_data = json.load(f)

    # 过滤无效数据
    ground_truth = [item for item in raw_data if item and item.get('query')]
    
    # 💡 调试模式：如果只想测前 10 条看看通不通，可以取消下面这行的注释
    # ground_truth = ground_truth[:10]

    print(f"🧪 Starting evaluation on {len(ground_truth)} queries...")

    results = []
    
    # --- 评测循环 ---
    for item in tqdm(ground_truth):
        query = item['query']
        target_img = item['image_filename'].strip() # 去除可能存在的空格

        try:
            # 执行检索：获取 Top 10 用于计算 Recall@10 和 MRR
            search_res = retriever.search_and_rerank(query, top_k_final=10)

            # 提取文件名 (标准化处理)
            retrieved_files = [os.path.basename(r['path']).strip() for r in search_res]

            # --- 计算指标 ---
            
            # 1. Recall@5
            top5_files = retrieved_files[:5]
            hit_5 = 1 if target_img in top5_files else 0
            
            # 2. Recall@10
            hit_10 = 1 if target_img in retrieved_files else 0
            
            # 3. MRR@10 (Mean Reciprocal Rank)
            # 如果在第 1 位，得 1 分；第 2 位，得 0.5 分... 没找到得 0 分
            mrr = 0.0
            if target_img in retrieved_files:
                rank = retrieved_files.index(target_img) + 1
                mrr = 1.0 / rank

            results.append({
                "id": item['id'],
                "recall@5": hit_5,
                "recall@10": hit_10,
                "mrr": mrr
            })

        except Exception as e:
            print(f"⚠️ Error processing query {item.get('id')}: {e}")

    # --- 统计输出 ---
    if results:
        df = pd.DataFrame(results)
        
        print(f"\n🏆 Benchmark Report for [{dataset_name}]")
        print("="*40)
        print(f"✅ Recall@5  : {df['recall@5'].mean():.4f}")
        print(f"✅ Recall@10 : {df['recall@10'].mean():.4f}")
        print(f"🥇 MRR       : {df['mrr'].mean():.4f}")
        print("="*40)
        
        # 保存详细结果到 CSV，方便分析 Bad Case
        output_csv = f"eval_results_{dataset_name}.csv"
        df.to_csv(output_csv, index=False)
        print(f"📄 Detailed results saved to {output_csv}")
        
    else:
        print("❌ No results generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., energy)")
    args = parser.parse_args()
    
    run_evaluation(args.dataset)
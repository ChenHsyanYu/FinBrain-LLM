import os
import pickle
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv

# ========== 初始化 ==========
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ========== 載入 Tokenizer ==========
def load_tokenizer():
    print("📥 載入 Tokenizer：taide/Llama-3.1-TAIDE-LX-8B-Chat")
    tokenizer = AutoTokenizer.from_pretrained(
        "taide/Llama-3.1-TAIDE-LX-8B-Chat",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ========== 建立通用語料快取 ==========
def build_general_cache(
    tokenizer, 
    dataset_name="liswei/Taiwan-Text-Excellence-2B",
    split="train",
    max_length=2048,
    cache_dir="./tokenized_cache",
    cache_filename="general_coct2B_2048.pkl",
    sample_ratio=1.0,  # 若要只取部分語料，可設 0.1 = 10%
):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        print(f"📂 已存在快取，跳過處理: {cache_path}")
        return cache_path

    print(f"🌏 下載 Hugging Face 語料: {dataset_name} ({split})")
    dataset = load_dataset(dataset_name, split=split)

    if sample_ratio < 1.0:
        sample_count = int(len(dataset) * sample_ratio)
        dataset = dataset.select(range(sample_count))
        print(f"✂️ 取樣比例: {sample_ratio*100:.1f}% ({sample_count} 筆樣本)")

    print(f"🧩 開始 Tokenize {len(dataset):,} 筆樣本（max_length={max_length}）...")

    tokenized_data = []
    for i, sample in enumerate(tqdm(dataset, total=len(dataset))):
        text = sample.get("text", "").strip()
        if not text:
            continue

        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        tokenized_data.append({
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0),
        })

        # 每 10k 筆顯示一次進度
        if (i + 1) % 10000 == 0:
            print(f"✅ 已完成 {i+1:,} 筆")

    # 儲存到本地
    cache_data = {"data": tokenized_data}
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)

    print(f"💾 已儲存通用語料快取 -> {cache_path}")
    print(f"📊 總筆數: {len(tokenized_data):,}")
    cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"📦 檔案大小: {cache_size_mb:.2f} MB")

    return cache_path


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    build_general_cache(tokenizer)

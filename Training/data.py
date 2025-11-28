import os
import json
import pickle
import s3fs
from pathlib import Path
from torch.utils.data import Dataset
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

class S3TextFileDataset(Dataset):
    def __init__(
        self, 
        s3_prefix: str, 
        tokenizer, 
        max_length: int = 2048,
        cache_dir: str = "./tokenized_cache",
        force_reprocess: bool = False
    ):
        """
        從 S3 讀取文字檔案並 tokenize，結果會快取到本地
        
        Args:
            s3_prefix: S3 路徑前綴 (例如: "s3://bucket/path/")
            tokenizer: HuggingFace tokenizer
            max_length: 最大 token 長度
            cache_dir: 本地快取目錄
            force_reprocess: 是否強制重新處理（忽略快取）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 建立快取檔案路徑（基於 s3_prefix 和 max_length）
        cache_key = f"{s3_prefix.replace('/', '_').replace(':', '_')}_{max_length}"
        self.cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # 如果快取存在且不強制重新處理，直接載入
        if self.cache_file.exists() and not force_reprocess:
            print(f"📦 從快取載入: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                cache_data = pickle.load(f)
            self.tokenized_data = cache_data["data"]
            self.file_list = cache_data["file_list"]
            print(f"✔️ 載入 {len(self.tokenized_data)} 筆資料（來自 {len(self.file_list)} 個檔案）")
            return
        
        # 否則從 S3 讀取並處理
        print("🔄 從 S3 讀取並 tokenize...")
        
        # 從環境變數讀取 AWS 憑證
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        if not aws_access_key or not aws_secret_key:
            raise ValueError("請在 .env 檔案中設定 AWS_ACCESS_KEY_ID 和 AWS_SECRET_ACCESS_KEY")
        
        self.fs = s3fs.S3FileSystem(
            key=aws_access_key,
            secret=aws_secret_key,
            client_kwargs={"region_name": aws_region},
        )
        
        if not self.fs.exists(s3_prefix):
            raise ConnectionError(f"無法存取 S3 prefix: {s3_prefix}")
        
        self.file_list = self._collect_files(s3_prefix)
        if not self.file_list:
            raise FileNotFoundError(f"在 {s3_prefix} 沒有找到支援的檔案 (.txt, .md, .jsonl)")
        
        print(f"✔️ 找到 {len(self.file_list)} 個檔案")
        
        # 預處理所有資料
        self.tokenized_data = self._preprocess_all()
        
        # 存檔
        print(f"💾 儲存到: {self.cache_file}")
        cache_data = {
            "data": self.tokenized_data,
            "file_list": self.file_list
        }
        with open(self.cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        
        cache_size_mb = self.cache_file.stat().st_size / (1024 * 1024)
        print(f"✔️ 儲存完成 ({cache_size_mb:.2f} MB)")
    
    def _collect_files(self, path):
        """使用 glob 收集 S3 路徑下所有支援的檔案"""
        collected = []
        
        print(f"🔍 開始掃描 S3 路徑: {path}")
        
        try:
            # 方法 1: 使用 glob 模式匹配（推薦，最快最可靠）
            # 確保路徑格式正確
            search_path = path.rstrip('/') + '/'
            
            # 分別搜尋三種檔案類型
            for pattern in ['**/*.txt', '**/*.md', '**/*.jsonl']:
                full_pattern = search_path + pattern
                print(f"   搜尋: {pattern}")
                files = self.fs.glob(full_pattern)
                collected.extend(files)
                print(f"   找到 {len(files)} 個檔案")
            
            # 去除可能的重複
            collected = list(set(collected))
        except Exception as e:
            print(f"❌ glob 搜尋失敗: {e}")
            print("   嘗試使用 find 方法...")
            
            try:
                # 方法 2: 使用 find（備用方案）
                all_files = self.fs.find(path)
                for file_path in all_files:
                    if file_path.endswith((".txt", ".md", ".jsonl")):
                        collected.append(file_path)
                        if len(collected) % 1000 == 0:
                            print(f"   已找到 {len(collected)} 個檔案...")
            except Exception as e2:
                print(f"❌ find 方法也失敗: {e2}")
                return []
        
        print(f"✔️  總共找到 {len(collected)} 個檔案")
        return collected
    
    def _read_file_content(self, file_path):
        """讀取 S3 檔案內容"""
        try:
            with self.fs.open(file_path, "r", encoding="utf-8") as f:
                if file_path.endswith(".jsonl"):
                    texts = []
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                text = obj.get("text") or obj.get("content", "")
                                if text:
                                    texts.append(text)
                            else:
                                texts.append(str(obj))
                        except json.JSONDecodeError:
                            continue
                    return "\n".join(texts)
                else:
                    return f.read()
        except Exception as e:
            print(f"⚠️  讀取錯誤: {file_path}, {e}")
            return ""
    
    def _preprocess_all(self):
        """預處理所有檔案並 tokenize"""
        tokenized_data = []
        total_files = len(self.file_list)
        
        for idx, file_path in enumerate(self.file_list):
            # 顯示進度
            if (idx + 1) % 10 == 0 or idx == 0 or idx == total_files - 1:
                print(f"處理進度: {idx + 1}/{total_files} ({(idx + 1) / total_files * 100:.1f}%)")
            
            text = self._read_file_content(file_path)
            if not text.strip():
                print(f"⚠️  跳過空檔案: {file_path}")
                continue
            
            tokenized = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
                add_special_tokens=True,
            )
            
            tokenized_data.append({
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "labels": tokenized["input_ids"].squeeze(0),
            })
        
        print(f"✔️ 成功處理 {len(tokenized_data)} 個檔案")
        return tokenized_data
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def clear_cache(self):
        """清除快取檔案"""
        if self.cache_file.exists():
            self.cache_file.unlink()
            print(f"🗑️  已刪除快取: {self.cache_file}")


# 使用範例
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
            "taide/Llama-3.1-TAIDE-LX-8B-Chat",
            trust_remote_code=True,
            token=HF_TOKEN,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 建立 dataset（第一次會從 S3 處理，之後會使用快取）
    dataset = S3TextFileDataset(
        s3_prefix="s3://fin-brain-nccu/clean_text",
        tokenizer=tokenizer,
        max_length=2048,
        cache_dir="./tokenized_cache",
        force_reprocess=False  # 設為 True 可強制重新處理
    )
    
    print(f"\n資料集大小: {len(dataset)}")
    
    # 取得第一筆資料
    sample = dataset[0]
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"attention_mask shape: {sample['attention_mask'].shape}")
    print(f"labels shape: {sample['labels'].shape}")
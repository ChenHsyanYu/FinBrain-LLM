import os
import json
import warnings
import torch
import torch.nn as nn
import botocore.exceptions
from tqdm.auto import tqdm
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
    TrainerCallback
)
from torch.utils.data import Dataset, random_split
import pickle
import math
# chsyu-national-chengchi-university/fin_brain_reset/z8afiu8i
os.environ["WANDB_PROJECT"]="fin_brain_reset"
os.environ["WANDB_RESUME"] = "allow"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["WANDB_RUN_ID"] = "z8afiu8i"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if "WANDB_RUN_ID" in os.environ:
        del os.environ["WANDB_RUN_ID"]

# ========== 初始化 ==========
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


# ========== 系統檢查 ==========
def check_system_resources():
    print("🔍 系統資源檢查:")
    print(f"   PyTorch 版本: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   GPU 數量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"   GPU {i} 記憶體: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB"
            )

        try:
            bf16_supported = torch.cuda.is_bf16_supported()
            print(f"   BF16: {'✅ 支援' if bf16_supported else '❌ 不支援'}")
        except:
            print("   BF16: ❓ 無法檢測")

        try:
            _ = torch.randn(10, device="cuda", dtype=torch.float16)
            print("   FP16: ✅ 支援")
        except:
            print("   FP16: ❌ 不支援")

        if hasattr(torch, "float8_e4m3fn"):
            try:
                _ = torch.randn(10, device="cuda").to(torch.float8_e4m3fn)
                print("   FP8: ✅ 支援")
            except Exception as e:
                print(f"   FP8 測試失敗: {e}")
        else:
            print("   FP8: 不支援")

        major, minor = torch.cuda.get_device_capability()
        print(f"   Compute Capability: {major}.{minor}")
    else:
        print("   僅 CPU 模式，建議 FP32")


# ========== 模型載入 ==========
def load_model_and_tokenizer():
    print("📥 載入 TAIDE 模型...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "taide/Llama-3.1-TAIDE-LX-8B-Chat",
            device_map=None,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "taide/Llama-3.1-TAIDE-LX-8B-Chat",
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ 模型載入成功")
        return model, tokenizer
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return None, None


# ========== Dataset ==========
class LocalTensorDataset(Dataset):
    def __init__(self, tensor_file: str, tokenizer=None, max_length: int = 2048):
        print(f"📦 載入快取檔案: {tensor_file}")
        with open(tensor_file, "rb") as f:
            self.data = pickle.load(f)

        self.tokenized_data = self.data["data"]
        print(f"✔️ 成功載入 {len(self.tokenized_data)} 筆樣本")

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        sample = self.tokenized_data[idx]
        return {
            "input_ids": sample["input_ids"].to(torch.long),
            "attention_mask": sample["attention_mask"].to(torch.long),
            "labels": sample["labels"].to(torch.long),
        }

# ========== 凍結參數 ==========
def freeze_model_layers(model, layer: str):
    for _, param in model.named_parameters():
        param.requires_grad = False

    if layer in ["last_transformer", "both"]:
        for name, param in model.named_parameters():
            if "model.layers.31." in name:
                param.requires_grad = True
    if layer in ["lm_head", "both"]:
        for name, param in model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"🔢 可訓練參數: {trainable:,} / {total:,} ({trainable/total:.2%})")


# ========== 訓練設定 ==========
def setup_training_args():
    name = "exp5_pure_finance"
    
    return TrainingArguments(
        output_dir=f"./models/{name}",
        logging_dir=f"./logs/{name}",
        run_name=name,
        
        # 訓練長度
        num_train_epochs=10,  # 增加訓練時間
        
        # 學習率設置 - 關鍵修復
        learning_rate=6e-5,
        lr_scheduler_type="cosine_with_restarts",  # 👈 改為恆定
        lr_scheduler_kwargs={"num_cycles": 2},
        warmup_steps=20,
        max_grad_norm=1.0,
        
        # 優化器
        optim="adamw_torch",
        adam_beta2=0.95,
        weight_decay=0.1,
        
        # 混合精度
        bf16=True,
        
        # Batch 設置
        per_device_train_batch_size=16,
        gradient_accumulation_steps=16,
        
        # 評估策略 - 重要添加
        eval_strategy="epoch",        # 👈 添加
        per_device_eval_batch_size=1,       # 更保守
        eval_accumulation_steps=1,          # 每步把暫存結果搬走，避免堆積
        prediction_loss_only=True, 
        
        # 保存策略
        save_strategy="epoch",
        load_best_model_at_end=True,  # 👈 添加（需要驗證集）
        metric_for_best_model="eval_loss",  # 👈 添加
        
        # 日誌
        logging_strategy="epoch",
        # logging_steps=52,  # 不需要每步都記錄
        
        # 其他
        gradient_checkpointing=False,
        dataloader_pin_memory=True,  # 建議改為 True
        dataloader_num_workers=0,    # 增加 workers
        remove_unused_columns=False,
        report_to="wandb",
    )


def compute_metrics(eval_pred):
    """回傳 perplexity，確保會被 log 到 W&B"""
    metrics = eval_pred.metrics
    if "eval_loss" in metrics:
        try:
            ppl = math.exp(metrics["eval_loss"])
        except OverflowError:
            ppl = float("inf")
        metrics["perplexity"] = ppl
    return metrics


class PerplexityCallback(TrainerCallback):
    """在 console 即時印出 perplexity"""
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            try:
                ppl = math.exp(metrics["eval_loss"])
            except OverflowError:
                ppl = float("inf")
            print(f"📈 Step {state.global_step}: eval_loss={metrics['eval_loss']:.4f} | ppl={ppl:.2f}")
        return control


def start_training(model, tokenizer, dataset, training_args):
    if not all([model, tokenizer, dataset]):
        print("❌ 模型、tokenizer 或資料集未就緒")
        return

    # ========== 凍結參數 ==========
    freeze_model_layers(model, "last_transformer")

    # ========== 載入通用語料並抽樣 ==========
    # general_path = "tokenized_cache/general_coct2B_2048.pkl"
    # all_samples = list(dataset)  # 先收集金融語料

    # if os.path.exists(general_path):
    #     print(f"\n📦 載入通用語料: {general_path}")
    #     with open(general_path, "rb") as f:
    #         general_data = pickle.load(f)
    #     general_samples = general_data["data"]
    #     total_general = len(general_samples)
    #     print(f"   通用語料總筆數: {total_general:,}")

    #     # ✨ 抽樣比例（3%）
    #     sample_ratio = 0.03
    #     sample_size = max(1, int(total_general * sample_ratio))
    #     print(f"   抽樣比例: {sample_ratio*100:.1f}% → 抽樣 {sample_size:,} 筆")

    #     import random
    #     random.seed(42)
    #     general_samples = random.sample(general_samples, sample_size)

    #     # 轉換成相同格式
    #     for s in general_samples:
    #         all_samples.append(
    #             {
    #                 "input_ids": s["input_ids"].to(torch.long),
    #                 "attention_mask": s["attention_mask"].to(torch.long),
    #                 "labels": s["labels"].to(torch.long),
    #             }
    #         )

    #     print(f"✅ 合併後資料總筆數: {len(all_samples):,}")
    # else:
    #     print("⚠️ 找不到 general_coct2B_2048.pkl，略過合併。")

    # ========== 合併後再切分 train / validation ==========
    print("📊 從混合語料切分訓練與驗證集...")
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    # 封裝成 Dataset 物件
    class MixedDataset(Dataset):
        def __init__(self, data_list):
            self.data = data_list
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    mixed_dataset = MixedDataset(dataset)
    train_dataset, val_dataset = random_split(
        mixed_dataset,
        [train_size, val_size],
        generator=torch.manual_seed(42),
    )

    print(f"   訓練集: {len(train_dataset):,} 筆")
    print(f"   驗證集: {len(val_dataset):,} 筆")

    # ========== 建立 Trainer ==========
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[PerplexityCallback()],
    )

    # ========== 開始訓練 ==========
    print("🔥 開始訓練...")
    result = trainer.train(resume_from_checkpoint=False)

    print("✅ 訓練完成")
    print(f"   最佳驗證 loss: {trainer.state.best_metric:.4f}")

    trainer.save_model()
    print("📁 模型已保存完畢。")
    return result




if __name__ == "__main__":
    check_system_resources()
    model, tokenizer = load_model_and_tokenizer()
    dataset = LocalTensorDataset("tokenized_cache/s3___fin-brain-nccu_clean_text_2048.pkl", tokenizer)
    training_args = setup_training_args()
    start_training(model, tokenizer, dataset, training_args)


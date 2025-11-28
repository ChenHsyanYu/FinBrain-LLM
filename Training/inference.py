# inference.py
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# ========= 1️⃣ 初始化 =========
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_PATH = "./models/exp5_pure_finance/checkpoint-364"   # 你自己的 fine-tuned 模型
BASE_MODEL = "taide/Llama-3.1-TAIDE-LX-8B-Chat"

# ========= 2️⃣ GPU 檢查 =========
print("🔍 檢查 GPU 狀態中...")
if torch.cuda.is_available():
    print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"🧮 精度: {dtype}")
else:
    print("⚠️ 未偵測到 GPU，將使用 CPU，速度可能會較慢。")
    dtype = torch.float32

# ========= 3️⃣ 載入模型與 tokenizer =========
print(f"📥 載入模型中: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_auth_token=HF_TOKEN,
    legacy=False,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=dtype,
    use_auth_token=HF_TOKEN,
)

# ========= 4️⃣ 定義生成函式 =========
def generate_response(system_prompt, user_prompt, max_new_tokens=512):
    """
    給定 system + user prompt，生成模型回答
    """
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 嘗試只保留 assistant 回答
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    return response

# ========= 5️⃣ 測試執行 =========
if __name__ == "__main__":
    print("\n💬 模型推論測試開始！\n")
    
    # 可自行修改這裡的 prompt 測試
    system_prompt = "你是一位專業的金融顧問，請以簡潔、生活化的方式解釋問題。"
    user_prompt = ""
    while True:
        user_prompt=input("請輸入您的問題：")
        output = generate_response(system_prompt, user_prompt)
        print("\n👤 使用者問題：", user_prompt)
        print("\n🤖 模型回覆：\n", output)
    
    
    

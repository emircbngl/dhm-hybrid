# DHM AI Asistanı — Fine-tune Kılavuzu

Bu dosya, lokal LLM asistanını lab-spesifik konuşmalarla özelleştirmek için
iki gerçek pipeline çıkarır. **Eğitim opsiyoneldir** — base model
(`qwen2.5:7b-instruct`) lab senaryolarında %90+ doğru tool çağrısı veriyor.
Fine-tune'u şu durumlarda düşün:

- Lab kendi kelime dağarcığını kullanıyor ("ASM yerine `prop2`", "10×
  objektif yerine `OB10x`") ve modelin bunları tool argümanlarına
  çevirmesini istiyorsun.
- Belirli zincirler tek prompt'la başlamalı: "günlük kontrol yap" →
  load + AF + recon + QPI + bundle export.
- Pilot kullanıcılar Türkçe sorduğunda asistanın Türkçe cevap vermesini,
  ama tool argümanlarını İngilizce/sayısal tutmasını istiyorsun.

---

## 0. Eğitim verisini üret

```bash
python scripts/ai_training_examples.py
# yazılan: data/ai/training_examples.jsonl  (12 örnek, 14 tool schema embed)
```

Üretilen dosya OpenAI fine-tune formatında — her satır bir konuşma:

```json
{"messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."},
              {"role":"assistant","tool_calls":[{...}]}, {"role":"tool","content":"..."}],
 "tools":[{"type":"function","function":{"name":"...","parameters":{...}}}, ...]}
```

**Lab-spesifik genişletme** için: `scripts/ai_training_examples.py:build_examples`
fonksiyonuna kendi senaryolarını ekle. Her örnek base'in zayıf olduğu yerleri
hedefle (yanlış z aralığı seçen, yanlış metric kullanan, Türkçe terim kullanan
prompt'ları yakala).

Hedef sayı: **50–200 örnek**. Daha az → underfit. Daha çok → over-narrow,
modelin genel yeteneği düşer.

---

## Pipeline A — Ollama Modelfile (en basit, eğitim YOK)

Base model'in üzerine **system prompt + few-shot örnekler** koyar; gerçek
weight güncelleme olmaz. 30 saniyede uygulanır, lab'da hemen test edilir.

```bash
# 1. Modelfile yaz (örnek)
cat > Modelfile <<'EOF'
FROM qwen2.5:7b-instruct

SYSTEM """You are the AI co-pilot for the Lindqvist Lab DHM platform. You
prefer Turkish replies but tool arguments stay in English/numeric. You know
the lab uses 532 nm and 10× / 40× objectives by default."""

PARAMETER temperature 0.2
PARAMETER num_ctx 8192

# Kısa, ölçülmüş örnekler (Modelfile MESSAGE direktifleri)
MESSAGE user "Günlük kontrol başlat"
MESSAGE assistant "Anladım — sırasıyla load, AF, recon ve QPI çalıştırıyorum."
EOF

# 2. Yerel modeli yarat
ollama create dhm-copilot -f Modelfile

# 3. AI panel ayarlarında modeli "dhm-copilot" yap; endpoint aynı kalır.
```

**Avantajı**: 30 saniye, GPU yok, geri alma trivial (`ollama rm dhm-copilot`).
**Dezavantajı**: Yeni davranış değil — sadece prompt-time bias.

---

## Pipeline B — HuggingFace `trl` ile gerçek LoRA fine-tune

Base model'in adapter weight'lerini eğit. ~4 GB VRAM'lık GPU veya MPS yeterli.
Apple Silicon'da çalışır (Mac Studio M2 Ultra: 50 örnek için ~15 dk).

### Kurulum

```bash
pip install transformers>=4.46 trl>=0.12 peft>=0.13 datasets>=3.0 accelerate
```

### Eğitim script'i (`scripts/finetune_lora.py`)

```python
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "data/ai/training_examples.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")

# Apple Silicon: device_map={"": "mps"} ekle. CUDA: "auto" yeter.
ds = load_dataset("json", data_files=DATA_PATH, split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds,
    args=SFTConfig(
        output_dir="out/dhm-copilot-lora",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,           # MPS için "fp16=True" yap
        dataset_text_field=None,   # SFTTrainer messages'ı otomatik render eder
        max_seq_length=4096,
    ),
    peft_config=LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none", task_type="CAUSAL_LM",
    ),
)
trainer.train()
trainer.save_model("out/dhm-copilot-lora/final")
```

### Ollama'ya import et

```bash
# 1. LoRA adapter'ı GGUF'a çevir (llama.cpp gerekli)
python llama.cpp/convert_lora_to_gguf.py out/dhm-copilot-lora/final \
    --outfile out/dhm-copilot-lora.gguf

# 2. Modelfile yaz
cat > Modelfile.lora <<'EOF'
FROM qwen2.5:7b-instruct
ADAPTER ./out/dhm-copilot-lora.gguf
PARAMETER temperature 0.2
EOF

ollama create dhm-copilot-tuned -f Modelfile.lora
```

AI panel ayarlarında `model_name` → `dhm-copilot-tuned`. Endpoint aynı.

**Maliyet (M2 Ultra)**: 50 örnek × 3 epoch ≈ 12 dk, 6 GB peak VRAM.
**Maliyet (CUDA L4)**: ≈ 4 dk.

---

## Eval — fine-tune'un işe yarayıp yaramadığını ölç

`tests/test_ai_finetune_eval.py` (henüz yok, eğitim yapacaksak yazarız) için
basit pattern:

```python
HOLDOUT = [
    ("Hologramı yükle ve odakla -25 ile +25 mm arasında",
     [("load_hologram",), ("find_focus_candidates",)]),
    ("Sample.tif'i aç recon at sonra QPI çıkar",
     [("load_hologram",), ("run_reconstruction",), ("run_qpi",)]),
    # 20-30 senaryo
]

def test_finetuned_calls_expected_tools(client):
    for prompt, expected_chain in HOLDOUT:
        events = list(agent.run(prompt, ...))
        called = [e.payload["call"].name for e in events
                  if e.kind == "tool_call_start"]
        for (expected,) in expected_chain:
            assert expected in called
```

**Başarı eşiği**: holdout'ta tool zinciri %95+ doğru. Bu altına düşerse
fine-tune verisini gözden geçir, üzerine çıkarsa ship.

---

## Hangisi senin için doğru?

| Senaryo | Pipeline |
|---|---|
| Yarın pilot demo, yeni davranış lazım | A (Modelfile) |
| Lab termsi öğretmek + Türkçe cevap | A (Modelfile, system prompt) |
| Tool zincirlerini tek-prompt'a indirmek | B (LoRA) |
| Spesifik metric/parameter seçim hatalarını düzeltmek | B (LoRA) |
| Base model zaten yetiyor | Hiçbiri — gerek yok |

Önerim: **önce A'yı dene (1 saat içinde test edebilirsin), holdout'ta tool
zinciri %85'in altında kalırsa B'ye geç**.

---

## Sprint 2 tool'ları — fine-tune verisinde mutlaka olmalı

`stage_focus_search`, `map_sample_grid`, `list_mapped_cells`, `goto_cell`,
`record_timelapse` — bu beş tool **lab-spesifik** kullanım için kritik
ve modelin doğal kullanması için eğitim verisinde örneklenmeli:

- **stage_focus_search**: "Yeni sample taktım, focus z'yi bilmiyorum" prompt'u →
  `search_range_mm=10` + `step_mm=0.5` + `refine_with_digital_af=true`. Model
  öğrenmesi gereken pattern: search range'i akıllı seç (ilk taramada geniş,
  yeniden ayar gereken durumda dar), her zaman digital AF ile bağla.
- **map_sample_grid**: "Sample üzerinde haritala" → grid bounds + sample_id
  zorunlu; cap ~6×6 (36 nokta) — daha büyük grid'lerde önce kullanıcıya sor.
- **goto_cell** ile **list_mapped_cells** zinciri: model önce listele, sonra
  cell_id seçtir. `nearest_to` argümanı operatör koordinat söylediğinde
  (haritalama UI'sından tıklayınca) tetiklenir.
- **record_timelapse**: `n_frames × interval_s` toplam süresini hesapla;
  10 dakika → 60 sn aralık → 10 frame, **30 dakikadan uzun** time-lapse'lar
  için kullanıcıya soru sor (lab kullanıcıları yanlışlıkla 1000 frame
  yazabilir).

## İlgili dosyalar

- [src/core/ai/tool_impls.py](../src/core/ai/tool_impls.py) — tool tanımları (eğitim verisinin gerçeği)
- [src/core/sample_map.py](../src/core/sample_map.py) — `SampleMap` + `CellLocation` (haritalama UI'sının data backbone'u)
- [scripts/ai_training_examples.py](../scripts/ai_training_examples.py) — JSONL üretici (17 örnek)
- [src/core/ai/agent.py](../src/core/ai/agent.py) — runtime loop, eval'in çalıştırdığı yer
- [src/gui/dialogs/ai_settings_dialog.py](../src/gui/dialogs/ai_settings_dialog.py) — model adı UI'sı

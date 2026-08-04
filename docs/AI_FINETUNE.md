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
# yazılan:
#   data/ai/training_examples.jsonl   (100 örnek, 11 aktif tool)
#   data/ai/eval_holdout.jsonl        ( 15 örnek, holdout — ASLA train'a koyma)
```

Üretilen dosya OpenAI fine-tune formatında — her satır bir konuşma:

```json
{"messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."},
              {"role":"assistant","tool_calls":[{...}]}, {"role":"tool","content":"..."}],
 "tools":[{"type":"function","function":{"name":"...","parameters":{...}}}, ...]}
```

### Lab profili (2026-Q2 baseline)

Training örnekleri şu konfigürasyona göre kalibre — değişirse
`scripts/ai_training_examples.py:LAB_PROFILE` sabitini güncelle ve regenerate et:

| Parametre | Değer |
|---|---|
| Işık kaynağı | HeNe λ=632.8 nm |
| Objektif | 50× air immersion (NA ~0.55–0.80) |
| Pixel pitch | 3.45 µm (Basler-class sensor) |
| Tipik z propagation | 0–15 mm |
| Sample inventory | USAF 1951, polystyrene bead, RBC, E. coli, Bacillus, Staph, Pseudo, Lacto |
| Operatör dili | Türkçe prose + İngilizce/sayısal tool args |

### Stage / device hardware askıda

Motorize stage + programlı shutter/LED henüz bağlı değil. Training
schema'sından **17 hardware tool'u dışlandı** (8 stage + 9 device).
Aktif tool sayısı **11**. Hardware geldikçe:

```bash
python scripts/ai_training_examples.py --include-stage              # motor takıldı
python scripts/ai_training_examples.py --include-stage --include-devices  # shutter/LED de
```

Lab-spesifik genişletme için `build_examples`'in çağırdığı 9 kategori
fonksiyonundan birine örnek ekle — dağılım hedefleri için
[AI_FINETUNE_DATA.md](AI_FINETUNE_DATA.md) §"8 Veri Kategorisi"'ye bak.

Hedef sayı: **50–200 örnek**. Daha az → underfit. Daha çok → over-narrow,
modelin genel yeteneği düşer. Mevcut: **100 train + 15 holdout = 115**.

---

## Pipeline A — Ollama Modelfile (en basit, eğitim YOK)

Base model'in üzerine **system prompt + few-shot örnekler** koyar; gerçek
weight güncelleme olmaz. 30 saniyede uygulanır, lab'da hemen test edilir.

Hazır Modelfile [Modelfile.dhm-copilot](../Modelfile.dhm-copilot)'da — lab profili
(HeNe 632.8 + 50× + sample inventory) zaten gömülü:

```bash
# 1. Base model'i indir (bir kez, ~4.7 GB)
ollama pull qwen2.5:7b-instruct

# 2. dhm-copilot tag'ini yarat
ollama create dhm-copilot -f Modelfile.dhm-copilot

# 3. AI panel ayarlarında model_name → "dhm-copilot"; endpoint aynı.

# Geri al:
ollama rm dhm-copilot
```

**Avantajı**: 30 saniye, GPU yok, geri alma trivial.
**Dezavantajı**: Yeni davranış değil — sadece prompt-time bias.

[Modelfile.dhm-copilot](../Modelfile.dhm-copilot) içeriği:
- SYSTEM prompt: lab profili + sample n_sample/n_medium tablosu + Türkçe yanıt kuralı
- 5 MESSAGE few-shot anchor: USAF kalibrasyon, RBC QPI, stage refusal, Cellpose refusal, range swap self-correction
- PARAMETER: `temperature=0.2`, `num_ctx=8192`, `top_p=0.9`

---

## Pipeline B — HuggingFace `trl` ile gerçek LoRA fine-tune

Base model'in adapter weight'lerini eğit. ~4 GB VRAM'lık GPU veya MPS yeterli.
Apple Silicon'da çalışır (Mac Studio M2 Ultra: 50 örnek için ~15 dk).

### Kurulum

```bash
pip install -r requirements_finetune.txt
```

İçeriği: `transformers>=4.46`, `trl>=0.12`, `peft>=0.13`, `datasets>=3.0`,
`accelerate>=1.0`. Detay: [requirements_finetune.txt](../requirements_finetune.txt).

### Eğitimi koş

Hazır script [scripts/finetune_lora.py](../scripts/finetune_lora.py)'de —
lab profili sabitleri + auto-detected backend (CUDA / MPS / CPU):

```bash
python scripts/finetune_lora.py
# → out/dhm-copilot-lora/final/  (LoRA adapter weights)
```

Default hyperparameter'lar (script'in başında belge edilmiş):
- LoRA `r=16`, `alpha=32`, `dropout=0.05`, target `q/k/v/o_proj`
- 3 epochs, batch=1, grad_accum=8, `lr=2e-4` cosine + 10% warmup
- `max_seq_length=4096`, MPS'de fp16 / CUDA'da bf16

Override etmek için CLI flag'ler: `--epochs`, `--lr`, `--lora-r`,
`--device`, `--out`. Tam liste için `--help`.

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

[tests/test_ai_finetune_eval.py](../tests/test_ai_finetune_eval.py) hazır —
holdout'tan 15 senaryoyu okur, modelden cevap ister, 4 metrik ölçer:

| Metrik | Eşik | Test fonksiyonu |
|---|---|---|
| Tool selection accuracy | ≥ 95 % | `test_tool_selection_accuracy` |
| Argument schema validity | ≥ 95 % * | `test_argument_schema_validity` |
| Refusal correctness | = 100 % | `test_refusal_correctness` |
| Chain-end has summary | ≥ 80 % | `test_chain_end_has_summary` |

\* AI_FINETUNE_DATA.md'de %98 yazıyor; biz holdout'a bilinçli olarak
self-correction case'leri (lowercase enum → server reject → düzeltme)
koyduğumuz için %95 daha realistic.

```bash
# Smoke (FakeLLMClient — referans cevabı replay eder; harness wiring testi)
pytest tests/test_ai_finetune_eval.py -v

# Real eval (lokal Ollama'ya karşı)
DHM_EVAL_LLM_ENDPOINT=http://localhost:11434 \
DHM_EVAL_LLM_MODEL=dhm-copilot \
pytest tests/test_ai_finetune_eval.py -v

# Yazdırılabilir rapor (pytest dışında)
DHM_EVAL_LLM_ENDPOINT=http://localhost:11434 \
DHM_EVAL_LLM_MODEL=dhm-copilot-tuned \
python tests/test_ai_finetune_eval.py
```

**Başarı eşiği**: dört metriğin hepsi yeşil. Biri düşerse fine-tune verisini
gözden geçir; arg validity %90'ın altına düşerse training data'da yapısal
bir hata var demektir (yanlış enum kullanan örnekler vs).

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

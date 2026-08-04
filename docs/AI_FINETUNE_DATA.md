# Fine-tune için Hangi Veriler Lazım?

Bu rehber **eğitim verisi tasarımı** için. Pipeline (LoRA / Modelfile) detayı
[AI_FINETUNE.md](AI_FINETUNE.md)'de. Burada *ne öğrettiğimizi* anlatıyoruz.

## TL;DR
- **Hedef**: lab-spesifik prompt → doğru tool zinciri + doğru argüman + doğru ton
- **Sayı**: 50–200 örnek MVP için yeter. 500+ olursa over-narrowing riski var
- **Her örneğin tipini etiketle** — aşağıdaki 8 kategorinin altında dağıtım yap
- **Holdout %10–20**: eğitime KOYMA, eval'de ölçüm için ayır
- **Veri kalitesi >> miktar**: 50 doğru örnek, 500 gürültülü örnekten iyi

---

## 8 Veri Kategorisi (her birinden en az 5 örnek)

### 1. Tool seçimi (which tool for this prompt)

Modelin doğal dilden tool'a eşlemeyi öğrenmesi.

```text
"Hologramı yükle"        → load_hologram
"Bu sample'ı haritala"   → map_sample_grid
"Recon çalıştır"         → run_reconstruction
"Şu an ne durumdayım?"   → get_state
"Bu hücreye git"         → goto_cell
"Geçen saat ne yaptım?"  → get_audit_tail
```

**Pattern**: 1 prompt → 1 tool çağrısı. Multi-step öğretmek için ayrı kategori (#3).

**Tuzak**: `set_recon_param` ile `run_reconstruction` arasında ayrım yapması gerek.
"λ=532 yap" sadece set, "λ=532 yap ve recon" set + run.

---

### 2. Argument formatting (sayısal + enum tipler)

Tool şemasının tam istediğini üretmek.

| Doğru | Yanlış |
|---|---|
| `wavelength_nm: 532` | `wavelength: "532 nm"` |
| `pixel_um: 5.0` | `pixel_um: "5e-6"` |
| `metric: "PHASE_VARIANCE"` | `metric: "phase_variance"` |
| `z_mm: 12.4` | `z_m: 0.0124` |
| `mode: "cell_aware"` | `mode: "cell-aware"` |

**5–10 örnekte** her sayısal birim ve her enum'u en az bir kere kullan.

**Negatif örnek**: kullanıcı "12.4 mm" yazdığında modelin `z_mm: 12.4` çıkarmasını
göster (string'i parse et, birim sayısallaştır).

---

### 3. Multi-tool zincirleri (chain reasoning)

Lab senaryolarının çoğu zincir: load → AF → recon → QPI.

**5–10 zincir örneği**, her biri 2–6 adım:

```text
Load → Find focus candidates → Set z → Recon → QPI    (yeni sample)
Load → Run autofocus → Recon → QPI                    (kalibre sample)
Stage focus search → Run autofocus → Recon            (z bilinmiyor)
Map sample → List cells → Goto cell → QPI             (haritada gezme)
Goto cell → Record timelapse                          (drift gözlem)
```

**Önemli**: zincirin son adımı **özet** olmalı — model "5 hücre QPI tamamlandı,
ortalama dry mass 6.4 pg" gibi sonuç yorumlamayı öğrensin. Sadece tool çıktısı
yapıştırma.

---

### 4. Self-correction (hata sonrası düzelme)

Tool error döndürünce model **aynı çağrıyı tekrarlamamalı**, argümanı düzeltmeli.

**Klasik düzeltme tipleri**:
- z_min > z_max → swap
- Tool yanlış → doğru tool'u dene
- Mode yanlış (cell_aware ama hücre yok) → auto'ya düş
- Path reddedildi (home dışı) → kullanıcıya neden olduğunu söyle, alternatif iste

**3–5 örnek** yeter — modelin "error gördüm, düşün, başka argüman dene" zincirini
öğrenmesi için.

---

### 5. Refusal & safety (reddetme)

Güvensiz / kapsam dışı talepleri reddetme.

```text
"/etc/passwd'i aç"          → reddet (path-traversal)
"Tüm settings'i sıfırla"    → reddet (irreversible, no tool)
"İnternete git ve X indir"  → reddet (no internet tool)
"Stage'i 999 mm'e gönder"   → reddet (clamp out of range)
```

**Pattern**: tool'u **ÇAĞIRMA**, açıklayıcı text yanıt ver. Veya tool'u çağır,
error sonucu açıkla.

**3 örnek** yeter — bu modelin doğal eğilimi zaten reddetmek, çok örnek
over-cautious yapar.

---

### 6. Domain language (lab termsi)

Türkçe ↔ İngilizce karışık prompt'larda doğru tool argümanı çıkarmak.

**Eşdeğer terimler** (training'de mutlaka her iki taraf görünmeli):
| Türkçe | İngilizce | Tool argümanı |
|---|---|---|
| numune, sample | sample | (sample_id, generic) |
| hücre, hücreler | cell, cells | (segmentation context) |
| odak, odaklama | focus, focusing | run_autofocus / stage_focus_search |
| derinlik, z derinliği | depth, z-depth | compute_depth_map / z_mm |
| dalga boyu | wavelength | wavelength_nm |
| büyütme | magnification | magnification |
| kuru kütle | dry mass | dry_mass_pg |

**Lab-spesifik kısaltmalar** (kullanıcılar gerçekten ne yazıyorsa):
- "AF" → autofocus
- "QPI" → run_qpi
- "OPD" → opd / phase
- "FFT" → fft_backend
- Lab özel kodlar — örn. "OB10x" → magnification: 10

**Strateji**: lab'da gerçekten kullanılan **5–10 prompt'u kayıttan al**, onları
örnek olarak kullan. Yapay senaryolar terimleri yanlış öğretir.

---

### 7. Conversational style (yanıt tarzı)

Modelin **kısa, action-first** yanıt vermesi için pozitif örnekler.

**İyi yanıt** (kısa, sayılı, aksiyon önerili):
> "Recon tamam — phase σ=1.21 rad, 0.78 s'de bitti. QPI çalıştırayım mı?"

**Kötü yanıt** (uzun, yorum yok, soru yok):
> "Reconstruction işlemi başarıyla tamamlandı. Pipeline'ın ürettiği sonuçlar
> mevcut ve görüntülenebilir. Sonraki adıma geçmek için ek talimat verebilirsiniz."

**Türkçe yanıt + İngilizce/sayısal tool args**: bunu öğretmek için 5–10 örneğin
**hepsinin** Türkçe assistant cevabı olsun, ama tool argümanları hep İngilizce
enum + numeric.

---

### 8. Negative examples (yapamadığı şeyler)

Modelin hayali tool'lar uydurmaması için.

```text
User: "Sample'ı renkli yap."
Assistant: "Renklendirme tool'um yok. Phase panel'de colormap değiştirmek
istersen: Tools menüsü → ... (UI üzerinden yapabilirsin)."
```

```text
User: "Cellpose ile segment et."
Assistant: "Cellpose entegrasyonu henüz yok (v3.0 roadmap'inde).
core.qpi.segment_cell_phase üzerinden basit threshold tabanlı segmentasyon
çalışıyor — onu kullanabilirim."
```

**3–5 örnek** — model "yapamam" demeyi de öğrenmeli, her şeye `{}` parametreli
random tool çağırmasın.

---

## Pratik öneriler

### Holdout split

Verinin %15–20'sini eval için ayır:

```bash
# eğitim: examples[:int(0.85*N)]
# eval:   examples[int(0.85*N):]
```

Eval'in **eğitime hiç değmemesi** kritik — yoksa fine-tune skoru kendini
kandırır.

### Veri toplama yolları

1. **Manuel** (en güvenilir): operator olarak 30 dakika kullan, gerçek
   prompt'larını + ideal cevaplarını yaz. 20 örnek çıkarsın yeter.
2. **Audit log madenciliği**: `~/.dhm-reconstruction/audit/*.jsonl` —
   gerçek tool çağrılarını + zamanlamaları gör. Hangi senaryolar sık?
3. **Self-distillation**: base model'e (qwen2.5:7b) 100 prompt sor,
   doğru cevap verdiklerini fine-tune verisi olarak kullan, yanlışları at.
4. **Synthetic varyasyon**: 1 prompt'un 3 farklı yazılışını üret (formal,
   casual, kısa). Modelin paraphrase'a karşı sağlamlığını artırır.

### Edge case'ler — eklemen gereken vakalar

| Edge case | Veri örneği |
|---|---|
| Boş state ("nothing loaded") | `get_state` sonrası "load et" önerisi |
| Çok büyük grid (>1000 nokta) | reddet veya kullanıcıya sor |
| Hücre yok ama cell_aware istendi | error → auto'ya düş |
| Path home dışı | reddet, alternatif iste |
| Mid-iteration cap (8 turdan fazla) | "iteration limit" yanıt et |
| Time-lapse > 30 dakika | önce kullanıcıya doğrula |
| Eski recon yok ama QPI istendi | önce recon çağır |

### Veri formatı (mutlaka)

OpenAI fine-tune chat formatı ([scripts/ai_training_examples.py](../scripts/ai_training_examples.py)
zaten bunu üretiyor):

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "tool_calls": [{...}]},
    {"role": "tool", "tool_call_id": "c1", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "tools": [...]
}
```

`tools` alanı her örnekte **canlı registry'den** gelmeli — registry değişirse
veri otomatik sync olsun.

### Eval kriterleri

Holdout üzerinde ölçülecek 4 metrik:

| Metrik | Eşik | Açıklama |
|---|---|---|
| Tool selection accuracy | ≥%95 | Doğru tool'u çağırıyor mu? |
| Argument schema validity | ≥%98 | JSON-Schema validate'i geçiyor mu? |
| Chain-end has summary | ≥%80 | Tool çıktısını yorumluyor mu? |
| Refusal correctness | %100 | Reddetmesi gereken yerde reddediyor mu? |

Bu 4'ünden biri %80'nin altına düşerse fine-tune **işe yaramamış** demektir,
veriyi gözden geçir.

---

## Toplam dağılım önerisi (100 örnek için)

| Kategori | Sayı |
|---|---|
| 1. Tool selection | 15 |
| 2. Argument formatting | 10 |
| 3. Multi-tool chains | 25 |
| 4. Self-correction | 8 |
| 5. Refusal & safety | 5 |
| 6. Domain language | 15 |
| 7. Conversational style | 12 |
| 8. Negative examples | 5 |
| Lab-spesifik (kayıttan) | 5 |
| **Toplam** | **100** |

**100 örnek = 1 günlük etiketleme** (öğleden sonra çekilecek bir iş).
**LoRA eğitimi M2 Ultra'da 100 örnek için ~25 dk**.

---

## Kayıt akışı

`scripts/ai_training_examples.py` 2026-04-28 itibarıyla **100 training + 15 holdout
örnek** üretiyor (lab profili: HeNe 632.8 nm + 50× obj + USAF/RBC/E.coli/Bacillus).
Stage tool'ları (8) ve device tool'ları (9) motorize hardware bağlanana kadar
training schema'sından dışlanıyor; aktif tool sayısı **11**.

Lab-spesifik veri eklemenin en temiz yolu:

1. `python scripts/ai_training_examples.py` ile JSONL'i üret (training + holdout).
2. `build_examples` çağırdığı 9 kategori fonksiyonundan birine yeni örnek ekle
   — `_chain_examples`, `_lab_specific_examples`, vb. Her fonksiyon kendi başına
   liste döndürür, dağılım kolayca dengelenir.
3. Tool registry değişirse (yeni tool eklenir / silinir / yeniden adlandırılır),
   otomatik sync olur — schema'lar canlı `build_tool_registry()` çağrısından
   embed ediliyor.

Hardware geri açma:
```bash
# Motorize stage takıldığında:
python scripts/ai_training_examples.py --include-stage
# Device hardware (shutter, LED, acquire_grid) bağlandığında:
python scripts/ai_training_examples.py --include-stage --include-devices
```

---

## Mevcut dağılım (2026-04-28 baseline)

| Kategori | Hedef | Mevcut | Notlar |
|---|---|---|---|
| 1. Tool selection | 15 | **15** | 11 aktif tool tek-call eşleme + set/run ayrımı |
| 2. Argument formatting | 10 | **10** | λ=632.8, M=50, n_sample/n_medium per sample |
| 3. Multi-tool chains | 25 | **25** | RBC, E. coli, Bacillus, USAF, bead full pipeline'lar |
| 4. Self-correction | 8 | **8** | range swap, enum case, prerequisite recovery |
| 5. Refusal & safety | 5 | **5** | path-traversal, internet, settings reset, stage |
| 6. Domain language | 15 | **15** | TR ↔ EN: dalga boyu, AF, kuru kütle, eritrosit, kok |
| 7. Conversational style | 12 | **12** | kısa + sayılı + sonraki adım önerili |
| 8. Negative | 5 | **5** | Cellpose, colormap, Excel, FastAPI, pyplot |
| 9. Lab-spesifik | 5 | **5** | USAF kalibrasyon, RBC dry mass, E. coli, bead, Bacillus |
| **Holdout (eval)** | 15 | **15** | ayrı dosya: `eval_holdout.jsonl` |
| **Toplam** | **115** | **115** | |

Stage hardware bağlanınca planlanan ek 15-20 örnek (`stage_focus_search`,
`map_sample_grid`, `goto_cell`, `record_timelapse` zincirleri) — hardware
gelene kadar askıda.

## İlgili dosyalar

- [scripts/ai_training_examples.py](../scripts/ai_training_examples.py) — JSONL üretici, 9 kategori fonksiyonu + holdout
- [scripts/finetune_lora.py](../scripts/finetune_lora.py) — Pipeline B (LoRA) eğitim script'i
- [Modelfile.dhm-copilot](../Modelfile.dhm-copilot) — Pipeline A (Ollama Modelfile)
- [tests/test_ai_finetune_eval.py](../tests/test_ai_finetune_eval.py) — holdout eval framework, 4 metrik
- [docs/AI_FINETUNE.md](AI_FINETUNE.md) — Pipeline kurulumu (A: Modelfile, B: LoRA)
- [src/core/ai/tool_impls.py](../src/core/ai/tool_impls.py) — tool schema'larının kaynağı

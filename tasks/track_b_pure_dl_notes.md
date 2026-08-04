# Track B — Pure Deep-Learning Reference-Free Reconstruction (Notes)

> **Durum**: aktif yol değil. Track C (hybrid CNN) tercih edildi 2026-05-05.
> Bu doküman Track B'nin **ileride** seçilebilmesi için ne gerekeceğinin
> referans notu. Mevcut data ile kantitatif olarak yetersiz olduğu kanıtlandı
> (bkz. `_benchmark_reffree/diagnose_fixed_z.png`).

---

## Track B nedir?

Tek bir off-axis hologram intensity'sinden, klasik fiziksel pipeline'a hiç
girmeden, doğrudan **unwrapped phase + amplitude** çıkaran end-to-end neural
network. Referans hologram yok, polynomial/Zernike fit yok, hatta
demodulation/propagation/unwrap kademelerinin hiçbiri DL içinde explicit
değil — net hepsini **örnek üzerinden** öğreniyor.

Mimari aileler:
- **eHoloNet** (Wang et al., 2018) — encoder-decoder CNN, raw hologram → phase
- **Y-Net** — tek hologramdan amp + phase paralel kollarda
- **HoloNet / HoloPhaseNet** (cGAN) — adversarial training, twin-image bastırma
- **MorpHoloNet** (2025, physics-driven) — fiziksel propagasyon backbone'a gömülü

## Bu setup için Track B uygulanabilir mi?

**Şu an: HAYIR.** Üç temel sebep:

1. **Eğitim veri ölçeği yetersiz**
   - Mevcut: 63 frame, 9 session
   - Track B literatürde tipik: **5,000–50,000 frame** (örn. eHoloNet ~10k synthetic + 1k real, Y-Net ~20k mixed)
   - 63 ile pure DL **kesinlikle** overfit eder ve generalize etmez

2. **Sample diversity düşük**
   - 9 session'ın çoğu aynı sample tipinde (USAF + bead karışık)
   - Pure DL "hangi numune olursa olsun" çıkarmayı öğrenmesi için sample-tipi
     diversite ZORUNLU (RBC + bakteri + bead + USAF + faz nesnesi … hepsi)

3. **Aberration footprint sabit**
   - Bu lab'da illumination beam profile + sensör fixed-pattern büyük ölçüde
     **session-stable** — dolayısıyla "fizik tarafı" basit
   - Pure DL'in tüm fiziği yeniden öğrenmesi gereksiz overhead; hybrid çok
     daha veri-verimli

## Track B'yi gerçekten denemek için minimum şart listesi

Eğer ileride bir pivot olur ve Track B gündeme gelirse:

### Veri (en kritik blok)
- [ ] **5,000+ real hologram** (8+ farklı sample tipi)
  - USAF (≥500), bead 3µm (≥500), bead 10µm (≥300)
  - RBC (≥500), E.coli (≥500), Bacillus (≥500)
  - Negative control / boş alan (≥500) — model öğrensin "buradaki nedir
    yoksa nedir değil" ayrımını
- [ ] Her real hologramın **doğrulanmış GT phase** olması — yani referansla
      kaydedilip, ref-divide + manual autofocus + manual unwrap ile temizlenmiş
- [ ] **20,000+ synthetic hologram** (training augmentation)
  - Numerical phantom: random RBC/bead distributions
  - Forward-model: angular spectrum + carrier + Gaussian beam + sensor
    Poisson/Gaussian noise + simulated dust speckle
  - GT trivially available (phantom = ground truth)
- [ ] Train/val/test splits: 80/10/10, **session-disjoint** (aynı session
      hem train hem test'te olamaz — leak)

### Mimari & training
- [ ] **U-Net (15-25M param)** baseline; raw 1024×1024 hologram → 1024×1024
      unwrapped phase + amplitude (2 channel output)
- [ ] **Sliding window inference** memory için — full frame'i tile et, overlap
      blend
- [ ] Loss = α·L1(phase) + β·L1(amp) + γ·L_TV (smoothness) + δ·SSIM
      + opsiyonel adversarial term
- [ ] Mixed-precision (FP16) training, A100 / RTX 4090 sınıfı GPU
- [ ] **2-3 hafta GPU saati** training time (5k epoch on 5k samples)

### Validation gates (Track A floor'u geçmeli)
| Metric | Target | Why |
|---|---|---|
| RMSE (vs validated GT) | <0.10 rad | hedefimizin altı |
| p95 abs err | <0.30 rad | local quality |
| Cross-session generalization | RMSE artışı <2× | leakage check |
| Out-of-distribution sample | not catastrophic | unseen sample tipi |
| Inference latency | <100 ms (1024² @ A100) | real-time gate |

### Engineering
- [ ] Model serving: TorchScript export, ONNX, vendor-specific (CoreML for
      M-series Mac inference)
- [ ] Drift monitoring: production'da p95 abs err'in canlı görselleştirmesi;
      eşik aşılınca operatöre uyarı + fallback (Track A poly5)
- [ ] Periyodik retrain: yeni session verisi geldikçe, lab bir buton
      tetikleyince transfer learning ile fine-tune

## Hybrid (Track C) → Track B'ye yumuşak geçiş yolu

Track C bir kez ürettikten sonra, ortaya çıkan hibrit pipeline kademeli olarak
"daha çok DL" tarafına ölçeklenebilir:

1. **C-1**: classical poly5 + small CNN residual corrector (current Track C)
2. **C-2**: classical demodulation + CNN aberration corrector + CNN unwrap
3. **B-naive**: end-to-end CNN, classical sadece demodulation
4. **B-full**: raw hologram → end-to-end output (klasik pipeline tamamen yok)

Her geçiş, önceki adımın validate edilmesine + yeni veriye dayanır. Track C
bittiğinde toplam veri +1000 frame seviyesine geldiyse C-2 düşünülebilir;
+5000'e geldiyse B-naive. Atlama yok.

## Referanslar

- eHoloNet — https://opg.optica.org/oe/abstract.cfm?uri=oe-26-18-22603
- HoloPhaseNet — https://pmc.ncbi.nlm.nih.gov/articles/PMC9352290/
- Y-Net — https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-15-21157
- MorpHoloNet (2025) — https://www.nature.com/articles/s41467-025-60200-x
- Phase recovery DL survey (Nature LSA 2023) — https://www.nature.com/articles/s41377-023-01340-x

## Karar günlüğü

- **2026-05-05**: Pure DL eleyildi. 63 frame yetersiz; structured-stripe
  residual classical floor'u 4 rad'e oturtuyor; bu pattern reproducible →
  Hybrid CNN (Track C) çok daha veri-verimli. Track B kapı açık ama önce
  veri toplama kampanyası gerekli.

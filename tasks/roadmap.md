# DHM Reconstruction — Combined Roadmap (v2.0.7 → v3.0)

**Son güncelleme:** 2026-04-27
**Durum:** v2.0.6 ship + 5-bug pilot patch + end-to-end + multi-focus refactor sonrası
**Test sayacı:** 518 pass / 1 skip / sıfır regresyon
**Bug regression:** 22 PASS / 0 FAIL / 8 LESSON / 5 MANUAL (`scripts/check_bugs.py`)

---

## Birleştirme Mantığı

Üç input'u birleştirdik:

1. **Lindqvist Lab toplantısı** (2026-04-27) — gerçek workflow pain'leri:
   time-lapse, drift, tracking, calibration, multi-user, paper-ready,
   GPU/headless, AI segmentation
2. **v2.0.7+ açık backlog** (CEO "hepsi" demeden gelmeyecek olanlar) —
   real hw camera, MP4, multi-line profile, batch resume, preset edit,
   audit viewer
3. **v2.0.3 + v2.0.6 review backlog** — drop zone live indicator, export
   highlight, line profile ROI, batch HDF5, crash handler, WCAG-AA,
   preset save/load
4. **Multi-focus refactor açtığı kapı** — batch FFT, GPU backend, ROI
   fast-path (3 perf fırsatı, hepsi `_make_fast_evaluator`'a düşüyor)

Her madde şu üç soruyla filtrelendi:

- Hangi version'da en doğal yer?
- Hangi dependency öncesinde gelmeli?
- Lab pressure'ı vs dev effort'u nereye düşüyor?

---

## Sürüm Tablosu

| version | tema | süre | deliverable |
|---------|------|------|-------------|
| **v2.0.7** | Time-lapse foundation | ~3 hafta | Session model + headless CLI + preset save/load + multi-user + per-frame CSV + audit viewer + batch resume |
| **v2.0.8** | Tracking & calibration | ~3 hafta | Drift correction + per-cell tracking + NIST bead calibration workflow + multi-line profile |
| **v2.0.9** | Paper-ready output | ~2 hafta | Vector PDF + Zenodo bundle + line profile ROI + crash handler + WCAG-AA + export highlight |
| **v2.1.0** | Performance — GPU + headless | ~4 hafta | PyTorch backend (CUDA/Metal) + batch FFT + ROI fast-path + Linux Docker + CI matrix |
| **v2.1.x** | Real hardware | ~3 hafta | Pylon/IDS/Thorlabs CameraSource + MP4 recording + live feed |
| **v3.0** | Faz 2 — AI | ~6-8 hafta | Cellpose + cell-cycle classifier + onboarding wizard |

---

## v2.0.7 — "Time-lapse foundation" (3 hafta)

### Hedef
Karin'in 3000-hologram pain'ini sıfırlamak. Pilot'tan production-grade
session ürününe geçişin omurgası.

### Sprint maddeleri

#### T0 — Time-series data model (4 gün)
- [ ] `src/core/session.py` — `Session` dataclass: `id`, `created_at`,
  `operator`, `sample_id`, `frames: List[HologramFrame]`,
  `params: ReconParams`
- [ ] `HologramFrame` — `path`, `timestamp_s`, `index`, `params_overrides`,
  `result: Optional[FrameResult]`
- [ ] `Session.from_directory(path, glob_pattern, sort_by="timestamp")` —
  ortak senaryo (lab klasörden zaman damgalı TIFF'leri yüklüyor)
- [ ] `Session.save_json()` / `Session.load_json()` — atomic write
- [ ] **Test:** `tests/test_session.py` — 10+ frame'den session, JSON
  roundtrip, missing frame error, mixed timestamps

#### T1 — Headless batch CLI (5 gün)
- [ ] `src/cli/run_session.py` — entry point: `python -m dhm.session run
  session.json --out results/`
- [ ] Phases: `load → preprocess → autofocus → reconstruct → qpi`
  ardışık veya `--phase autofocus` filtreli
- [ ] Progress: stdout JSONL line per frame (tail -f friendly)
- [ ] Cancel: SIGINT → graceful, partial results saved
- [ ] `--workers N` paralel frame processing (multiprocessing pool)
- [ ] **Test:** `tests/test_cli_session.py` — synthetic 5-frame session,
  CLI çağırılır, output dir doğru shape, SIGINT graceful

#### T2 — Per-frame CSV export (2 gün)
- [ ] `src/core/session_export.py::write_session_csv` —
  hücre × zaman matrisi: `frame_idx, timestamp_s, cell_id, cy, cx,
  z_mm, dry_mass_pg, area_um2, height_nm`
- [ ] Hem long format (her satır bir hücre × frame) hem wide format
  (`--format wide` → satır frame, sütun cell_X_dry_mass)
- [ ] **Test:** `tests/test_session_export.py` — fake session, both
  formats, CSV header doğru

#### T3 — Multi-user profile (3 gün)
- [ ] `src/core/user_profile.py` — `current_user()` (env / OS),
  `~/.dhm-reconstruction/users/<username>/state.json` ayrımı
- [ ] `audit_log.record(...)` artık `operator=current_user()` field
  embed
- [ ] Migration: tek kullanıcılı state otomatik default user'a taşınır
- [ ] **Test:** `tests/test_user_profile.py` — env var override, state
  isolation, audit operator field

#### T4 — Preset save/load (2 gün) [v2.0.3 backlog'undan gelir]
- [ ] `src/core/preset_store.py` — kullanıcı dict'i,
  `~/.dhm-reconstruction/users/<u>/presets.json`
- [ ] UI: sidebar preset combo'nun yanında "+" / "Edit existing" butonu
  (audit log'da kayıt — edit eski versiyonu archive eder)
- [ ] Preset'ler artık (built-in) + (user) iki source — built-in salt-okunur
- [ ] **Test:** `tests/test_preset_store.py` — save, load, edit
  (overwrite + archive), roundtrip

#### T5 — Audit log viewer (2 gün) [v2.0.7+ backlog'undan]
- [ ] `src/ui2/dialogs/audit_viewer.py` — DPG window, JSONL parse,
  filterable table (action, operator, time)
- [ ] Help → "Show audit log..." menu item
- [ ] **Test:** `tests/test_audit_viewer.py` — fake JSONL, filter
  semantics

#### T6 — Batch resume (2 gün) [v2.0.7+ backlog'undan]
- [ ] `Session.compute_signature()` — params + frame paths + sizes hash
- [ ] CLI `--resume-if-exists`: aynı signature için önceki output dir
  varsa frame'leri tek tek skip et
- [ ] **Test:** `tests/test_session_resume.py` — partial output dir,
  resume completes only missing frames

#### Sprint sonu beklenen
- 518 + ~50 test = ~568 pass
- `python -m dhm.session run pilot_session.json --out /tmp/out`
  CLI komutu Linux/Mac'te çalışıyor
- Karin gerçek 3000-hologram session'ını overnight koyup CSV alabilir
- Audit log per-operator filtreliyor, multi-user state ayrılmış

---

## v2.0.8 — "Tracking & calibration" (3 hafta)

### Hedef
Karin'in 1. ve 3. pain madde'leri (drift + calibration), Anna'nın
"trust the numbers" baskısı.

### Sprint maddeleri

#### Drift correction (5 gün)
- [ ] `src/core/registration.py::estimate_drift_phase_correlation` —
  scipy.fft.fft2 ile ardışık frame'ler arası shift estimate
- [ ] Session pipeline opsiyonel `--register-drift` ile drift_yx'i
  her frame'e ekler
- [ ] Drift > X µm uyarı (sample slide off frame)
- [ ] **Test:** synthetic drift inject, recover within ±1px

#### Per-cell tracking (4 gün)
- [ ] `src/core/tracking.py` — depth_map cluster centroidleri →
  `trackpy.link_df` ile cell_id stable across frames
- [ ] Cell birth / death events: opsiyonel JSON event log
- [ ] **Test:** synthetic 3-cell time series, ID stability

#### NIST calibration workflow (4 gün)
- [ ] Tools → "Calibration check" dialog: NIST 10µm bead hologram yükle
  → tool measure et → bilinen 10µm ile compare → drift % raporla
- [ ] `~/.dhm-reconstruction/users/<u>/calibration_history.jsonl`
  (date, measured, drift_percent, operator)
- [ ] Drift > 5% → red, 2-5% → yellow, < 2% → green
- [ ] **Test:** synthetic 10µm bead, drift detect

#### Multi-line profile (2 gün) [v2.0.7+ backlog'undan]
- [ ] Profile dialog: N tane click-drag çizgisi, her birine renk + label
- [ ] Profile karşılaştırma overlay
- [ ] **Test:** synthetic phase, 3 lines → 3 traces

---

## v2.0.9 — "Paper-ready output" (2 hafta)

### Hedef
Sven'in B maddesi (Nature/EMBO formatı). Lab paper'ı çıkarken DHM
verisini citation-ready bundle olarak teslim edebilmek.

### Sprint maddeleri

- [ ] **Vector PDF export** — matplotlib backend (DPG raster yerine);
  scale bar + colorbar + ticks + legend; `dpi=300` default
- [ ] **Zenodo-ready bundle** — `bundle.zip`: `figures.pdf`,
  `raw_data.csv`, `params.json`, `checksum.txt`, `README.md`
- [ ] **Line profile click-drag ROI** [v2.0.6 review'dan] — center row
  yerine elle çizilebilir ROI
- [ ] **Crash handler** [v2.0.6 review'dan] — `sys.excepthook` →
  toast + audit log + safe-state save
- [ ] **High-contrast theme WCAG-AA test** [v2.0.6 review'dan] — palette
  zaten var, programatik contrast ratio test
- [ ] **Report mode export buttons highlight** [v2.0.3 backlog'dan] —
  workflow=Report olduğunda "Export" buttons accent rengiyle yan-vurgu
- [ ] **Drop zone reconstruction live indicator** [v2.0.3 backlog'dan] —
  status text yerine üstte progress bar (Karin'in batch çalışırken
  ilerlemeyi gösteren ek görsel)

---

## v2.1.0 — "Performance: GPU + headless" (4 hafta)

### Hedef
Sven'in C maddesi (production gate) + multi-focus refactor'ün açtığı
3 perf fırsatını çakmak.

### Sprint maddeleri

#### PyTorch backend for `_make_fast_evaluator`
- [ ] `src/core/fft_backend_torch.py` — torch.fft.fft2 wrapper,
  CUDA / Metal / CPU otomatik
- [ ] Backward-compat: `get_best_fft_backend()` yeni torch backend'i
  prefer eder eğer CUDA/Metal varsa, fallback np.fft
- [ ] `_make_fast_evaluator` agnostik kalır — backend swap üzerinden
  hem autofocus 6 algo hem multi-focus otomatik kazanır
- [ ] **Bench hedef:** 2048² × 40 step zscan
  - Şu an Mac CPU: 4.3 sec
  - Hedef Mac MPS: <1 sec (5×)
  - Hedef CUDA RTX 4090: <0.3 sec (15×)

#### Batch FFT
- [ ] `evaluator.batch_evaluate(zs)` — N transfer kernel × tek field
  spectrum → tek batched ifft (torch.fft.ifft2 batched)
- [ ] M-series Mac'te ~2× kazanç bench hedefi (yan tek call dispatch
  overhead'i amorize ediyor)

#### ROI fast-path
- [ ] `_make_fast_evaluator` ROI verildiyse: ROI dışındaki bin'leri
  sıfırla, FFT shape değişmez ama compute yarıdan fazlası boş
- [ ] Büyük frame küçük ROI senaryosunda 4× kazanç bench hedefi

#### Headless infra
- [ ] CI matrix: GitHub Actions Linux + Mac + Windows (DPG headless
  mode kontrol)
- [ ] Linux Docker image (NVIDIA base + venv)
- [ ] Cluster compute deployment guide (Sven'in IT ekibine doc)

---

## v2.1.x — "Real hardware" (~3 hafta)

CEO + Anna onayıyla sırası gelirse:

- [ ] **Pylon CameraSource** (Basler kameralar, lab'da ana kullanım)
- [ ] **IDS uEye** ve **Thorlabs SciCam** Protocol implementations
- [ ] **MP4 recording** [v2.0.7+ backlog] — imageio-ffmpeg, live feed'i
  canlı kaydet
- [ ] Live preview: 30fps reconstruction preview, parametreler değiştikçe
  güncel
- [ ] **Test:** mock camera fixture, recording sigortası

---

## v3.0 — "Faz 2: AI segmentation" (6-8 hafta)

Erik'in onboarding pain'i + Anna'nın stratejik vizyonu.

- [ ] **Cellpose entegrasyonu** — pretrained model, per-cell mask
  out of box
- [ ] **Cell-cycle classifier** — dry_mass + morphology features →
  G1/S/G2/M (training data Karolinska'dan)
- [ ] **Onboarding wizard** — sample tipi seç (A549/HeLa/RBC/custom) →
  preset auto-load → first hologram bilinen ground truth karşı tool
  default'larını sanity-check
- [ ] **Licensing** — Faz 2'nin ikinci ayağı; ipuçları Anna toplantısında
  bahsedilmedi ama CLAUDE.md'de var

---

## Faz 2 Backlog (Henüz scheduled değil)

- **PDF generation pipeline** — Faz 2 ana hatlarından
- **Licensing** — Faz 2 ana hatlarından

---

## User-blocked verifyer (sen müsait olunca)

- [ ] **B-029 (Bug #4):** gerçek senaryo — hangi shape (1024²/2048²?),
  hangi algoritma, n_steps kaç? `scripts/bench_autofocus.py
  --shapes <senin>` ile kesin nokta
- [ ] **B-026 (Bug #1):** flip yönü — gerçek TIFF'inde Vertical mi
  Horizontal mı doğru? Sensör hep aynı yöndeyse default'a alırız
- [ ] **B-030 (Bug #5):** scroll — 1440×800 MacBook Air'de tier 288
  bekleniyor, scroll olmamalı

---

## Bağımlılık grafiği

```
v2.0.7 Time-lapse foundation
   ├─ v2.0.8 Tracking + calibration  (session model'e dayanıyor)
   ├─ v2.0.9 Paper-ready output      (session.export → bundle.zip)
   ├─ v2.1.0 GPU + headless          (CLI'a backend swap)
   │       └─ v2.1.x Real hardware   (live feed CLI üstünden)
   └─ v3.0  AI segmentation          (per-frame mask → session entry)
```

`v2.0.7` her şeyin omurgası. **Önce o ship olur**, paralel branch yok.

---

## Sprint cycle ritüeli

Her sprint sonunda:

1. `pytest tests/ -q` → 0 fail, 0 regression
2. `python scripts/check_bugs.py` → tüm phase'lerde 0 FAIL
3. `python scripts/check_bugs_phase_<current>.py` → bu sprint'in
   bug'ları yeşil
4. `python scripts/bench_autofocus.py` → trend tablosu update
5. `tasks/lessons.md` → bu sprint'in dersleri
6. `tasks/roadmap.md` → bu doc, "Şu anki version" satırı update
7. Lab demo: Karin / Sven / Erik 30-dk hands-on; "phase X yeşil mi?"
   sorusuna `check_bugs_phase_X.py` 1 saniyede yanıt verir

**Yeni phase başlatmak**:
1. `scripts/bug_registry.py` → `Phase` enum'una yeni member.
2. `scripts/check_bugs_phase_<key>.py` → 5-satırlık wrapper (mevcut
   örneklerden kopyala). Wrapper boş phase için "no bugs registered
   yet" banner'ı bastırır → exit 0.
3. Sprint içinde yakalanan her regresyonu **yakaladığın gün**
   registry'ye phase tag'iyle ekle. Sprint sonu toplu giriş =
   tarih sapması.

**Bug-history aracı yapı**:
* `scripts/bug_registry.py` — data only (`BugEntry`, `BUG_REGISTRY`,
  `Phase` enum, `entries_for_phase()`)
* `scripts/_bug_runner.py` — runner + formatter (paylaşılan)
* `scripts/check_bugs.py` — tüm phase + `--phase` filter
* `scripts/check_bugs_phase_*.py` — 10 wrapper, 5 aktif + 5 future

`tasks/lessons.md § 2026-04-27` bu kuralın gerekçesini saklar.

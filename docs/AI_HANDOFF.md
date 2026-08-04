# AI Modülü — Frontend Devir Paketi

Bu dosya **frontend ekibine devir** içindir. Backend (core + worker + base panel)
ship edildi. Frontend'in göreceği yüzey, data shape'leri, mevcut entegrasyon
noktaları ve sıradaki UI işleri burada.

---

## 0. TL;DR — Frontend ne yapıyor?

Backend bitti: AI agent local LLM'e bağlanıp 28 tool çağırıyor (pipeline +
mapping + time-lapse + device kontrolü). Mevcut **chat paneli** var. Frontend
bunun **üzerine bina edecek**:

1. **Sample map görselleştirme** — `SampleMap` → 2-D scatter overlay, hücre
   tıkla → `goto_cell`
2. **Device control panel** — shutter / LED / stage manuel sürme butonları
   (linter tool'ları zaten hazır)
3. **Tool-call inspector** — AI'ın geçmişte ne çağırdığını replay
4. **Time-lapse player** — `record_timelapse` çıktılarını oynatma
5. **Stage position HUD** — gerçek zamanlı XYZ status bar

---

## 1. Mimari (text diagram)

```
                  ┌──────────────────────────────────────────────┐
                  │            QMainWindow (gui/main_window)     │
                  │                                              │
   ┌──────────────┴───────┐  ┌─────────────────┐  ┌──────────────┘
   │ Sidebar (recon/AF/   │  │ Image grid      │  │ AI dock (right)
   │ QPI tabs)            │  │ (input/amp/     │  │ ┌─────────┐
   │                      │  │  phase/spec)    │  │ │AIPanel  │
   └──────────────────────┘  └─────────────────┘  │ │  chat   │
                                                  │ │ history │
                                                  │ │ ─────── │
                                                  │ │ input   │
                                                  │ │ Send/Stop│
                                                  │ └────┬────┘
                                                  └──────┼──────┐
                                                         │      │
                                            ┌────────────▼──┐   │
                                            │ AIWorker(Qt   │   │
                                            │  Thread)      │   │
                                            └────────┬──────┘   │
                                                     │          │
                              ┌──────────────────────┴────────┐ │
                              │ Agent loop: chat→tool→loop    │ │
                              │ (core/ai/agent.py)            │ │
                              └─────────┬─────────────────────┘ │
                                        │                       │
       ┌────────────────────────────────┼─────────────────────┐ │
       │ ToolRegistry — 28 tools        │                     │ │
       │ (core/ai/tool_impls.py)        │                     │ │
       └─┬──────────────────────────────┼──────┬──────────────┘ │
         │                              │      │                │
         │ AI-thread tools              │      │ GUI-thread tools (marshalled)
         │ (state, audit, stage, segment│      │ (load/recon/AF/QPI/depth/capture)
         │  map, timelapse loop)        │      │
         ▼                              ▼      ▼                │
   ┌──────────┐  ┌──────────┐  ┌──────────────────────────────┐ │
   │MockStage │  │SampleMap │  │ MainWindow._trigger_*        │ │
   │ (XYZ)    │  │ (cells)  │  │   ↓                          │ │
   └──────────┘  └──────────┘  │ Existing pipeline workers    │ │
                               │ (recon/AF/QPI QThreads)      │ │
                               └──────────────────────────────┘ │
                                                                │
   ┌────────────────────────────────────────────────────────────┘
   │ LocalLLMClient (HTTP) → Ollama / LM Studio
   │ (core/ai/client.py)
   └────────────────────────────────────────────────────────────┘
```

---

## 2. Dosya envanteri

### Yeni — core (saf Python, Qt yok)
| Dosya | Amaç |
|---|---|
| [src/core/ai/__init__.py](../src/core/ai/__init__.py) | Public re-export |
| [src/core/ai/protocol.py](../src/core/ai/protocol.py) | `ChatMessage`, `ToolCall`, `AssistantTurn` dataclass'ları |
| [src/core/ai/client.py](../src/core/ai/client.py) | `LocalLLMClient` — Ollama + OpenAI compat sync HTTP |
| [src/core/ai/tools.py](../src/core/ai/tools.py) | `ToolRegistry`, `ToolSpec`, `ToolContext` |
| [src/core/ai/tool_impls.py](../src/core/ai/tool_impls.py) | 28 tool'un implementasyonu + factory `build_tool_registry()` |
| [src/core/ai/context.py](../src/core/ai/context.py) | `StateSnapshot` + sistem prompt builder |
| [src/core/ai/agent.py](../src/core/ai/agent.py) | Chat→tool→loop, iteration cap, cancel |
| [src/core/ai/security.py](../src/core/ai/security.py) | `validate_path`, `clamp`, `redact_for_llm`, `NUMERIC_BOUNDS` |
| [src/core/stage.py](../src/core/stage.py) | `StageInterface` ABC + `MockStage` (gerçek hardware'a swap noktası) |
| [src/core/sample_map.py](../src/core/sample_map.py) | `CellLocation` + `SampleMap` (JSON persist, nearest query) |

### Yeni — GUI
| Dosya | Amaç |
|---|---|
| [src/gui/workers/ai_worker.py](../src/gui/workers/ai_worker.py) | `AIWorker(QThread)` — bir agent run'ı, Qt signal'leri |
| [src/gui/panels/ai_panel.py](../src/gui/panels/ai_panel.py) | `AIPanel(QDockWidget)` — chat UI, tool call rendering |
| [src/gui/dialogs/ai_settings_dialog.py](../src/gui/dialogs/ai_settings_dialog.py) | `AISettingsDialog` — endpoint/model/temp ayarları |
| [src/ui2/device_panel.py](../src/ui2/device_panel.py) | **v2.2.0-devices** — Stage HUD + shutter/LED/jog controls (DPG; W2 + W5) |

### Yeni — script + doküman
| Dosya | Amaç |
|---|---|
| [scripts/ai_training_examples.py](../scripts/ai_training_examples.py) | Fine-tune JSONL üretici (20 seed örnek) |
| [docs/AI_FINETUNE.md](AI_FINETUNE.md) | Pipeline (Modelfile / LoRA) kurulum kılavuzu |
| [docs/AI_FINETUNE_DATA.md](AI_FINETUNE_DATA.md) | Veri tasarımı: hangi kategoriden kaç örnek |

### Modifiye — mevcut dosyalar
| Dosya | Değişiklik |
|---|---|
| [src/core/settings_schema.py](../src/core/settings_schema.py) | `AIDefaults` dataclass, `AppSettings.ai`, `with_ai()`, `SCHEMA_VERSION = 12` |
| [src/gui/settings_store.py](../src/gui/settings_store.py) | `_migrate_v11_to_v12` + ai/* key okuma/yazma |
| [src/gui/main_window.py](../src/gui/main_window.py) | `_install_ai_panel`, `_toggle_ai_panel`, cache push hook'ları |
| [src/gui/commands_install.py](../src/gui/commands_install.py) | `ai.toggle_panel` (Cmd+Shift+A) komutu |
| [requirements.txt](../requirements.txt) | `requests>=2.31`, `jsonschema>=4.21` |

### Testler (133 test)
`tests/test_ai_{client,tools,tools_advanced,agent,context,security,panel,examples_emit}.py`,
`tests/test_stage.py`, `tests/test_sample_map.py`, `tests/test_settings_schema_ai.py`

---

## 3. Tool envanteri — 28 tool

`build_tool_registry(include_devices=True)` (default) → tümü
`build_tool_registry(include_devices=False)` → kanonik 19

### Pipeline (7)
| # | Tool | Kullanım |
|---|---|---|
| 1 | `load_hologram` | Hologram dosyası aç (path validation) |
| 2 | `set_recon_param` | Sidebar param güncelle (wavelength, pixel, z, n_medium, mask, …) |
| 3 | `run_reconstruction` | Off-axis pipeline çalıştır |
| 4 | `run_autofocus` | Z aralığında en iyi z'yi bul |
| 5 | `find_focus_candidates` | Multi-focus — top-K aday |
| 6 | `run_qpi` | Quantitative Phase Imaging (OPD, dry mass, …) |
| 7 | `compute_depth_map` | Per-pixel best-focus z |

### State / introspection (3)
| # | Tool | Kullanım |
|---|---|---|
| 8 | `get_state` | Loaded path, current params, last result summaries |
| 9 | `get_last_result` | `{stage: recon|af|qpi|depth}` |
| 10 | `get_audit_tail` | Son N audit log entry'si (operator redact) |

### Stage (4)
| # | Tool | Kullanım |
|---|---|---|
| 11 | `stage_get_position` | XYZ mm |
| 12 | `stage_move_relative` | dx/dy/dz delta hareket |
| 13 | `stage_move_absolute` | Mutlak XYZ |
| 14 | `stage_home` | (0,0,0)'a dön |

### Sprint 2 — focus search + mapping + timelapse (5)
| # | Tool | Kullanım |
|---|---|---|
| 15 | `stage_focus_search` | Z stage sweep + sharpness peak. **Mode: auto / cell_aware / full_frame**. `mask_dilate_px` ile mask büyüt. Optional digital AF refine. |
| 16 | `map_sample_grid` | XY grid sweep → her noktada segment et → `SampleMap`'a kaydet. ≤1024 nokta cap. |
| 17 | `list_mapped_cells` | Map'teki hücreleri dök (`limit` ile sınırla) |
| 18 | `goto_cell` | `cell_id` veya `nearest_to:{x,y}` ile haritalı hücreye git |
| 19 | `record_timelapse` | N frame × interval_s, opsiyonel recon/QPI per frame |

### Linter device tools — APT-uyumlu (9)
| # | Tool | Kullanım |
|---|---|---|
| 20 | `list_devices` | Mevcut shutter/LED bağlı mı |
| 21 | `shutter_open` | Shutter aç |
| 22 | `shutter_close` | Shutter kapat |
| 23 | `shutter_status` | Aç/kapalı durumu |
| 24 | `led_set_intensity` | `intensity_percent: 0-100` |
| 25 | `led_on` | LED aç (önceki intensity) |
| 26 | `led_off` | LED kapat |
| 27 | `led_status` | Durum + intensity |
| 28 | `acquire_grid` | Tek shot'ta shutter+LED+grid (rows × cols, spacing_um) |

> **Not**: Device tool'ları `ToolContext.shutter` / `.led` / `.orchestrator`
> bind edilmediği sürece `{"error": "<kind> device not configured"}` döndürür —
> hardware yokken bile crash etmez.

---

## 4. Frontend'in tüketeceği public API'ler

### a) `core.sample_map.SampleMap` — Map görselleştirme

```python
from core.sample_map import SampleMap, CellLocation
from core.user_profile import user_state_dir

# Disk yolu (panel zaten yazıyor, frontend okuyor):
path = user_state_dir() / "sample_maps" / f"{sample_id}.json"
sm = SampleMap.load(path)

sm.cells              # list[CellLocation]   — scatter için
sm.grid_extent_mm     # (x_min, x_max, y_min, y_max) — eksen sınırları
sm.grid_step_mm       # float — grid çözünürlüğü
sm.created_at         # ISO timestamp
sm.sample_id          # string ID

# Query:
sm.by_id(cell_id)              # CellLocation | None
sm.nearest(x_mm, y_mm)         # En yakın hücre
sm.summary()                   # {count, grid_extent_mm, …}

# CellLocation alanları:
# cell_id, stage_x_mm, stage_y_mm, stage_z_mm,
# in_frame_y_px, in_frame_x_px,    # hologram içi pixel
# area_um2, dry_mass_pg, notes
```

**Live update**: `AIPanel._sample_map` instance referansı. Panel `map_sample_grid`
bittikten sonra `_persist_sample_map()` ile diske yazıyor. Frontend `QFileSystemWatcher`
ile dosyayı izleyebilir.

### b) `core.stage.MockStage` — Live stage position

```python
from core.stage import StageInterface

stage = main_window._ai_panel.stage()    # AIPanel.stage()
x, y, z = stage.get_position()           # tuple[float, float, float]

# Listener — pozisyon değişince tetiklenir:
unsubscribe = stage.add_listener(lambda pos: print("moved to", pos))
# ...
unsubscribe()
```

Frontend bunu **status bar'a** veya **device control panel'e** bağlayıp anlık
pozisyon gösterebilir. Hardware'a geçince swap drop-in (`StageInterface` ABC).

### c) `core.ai.tools.ToolRegistry` — Tool inspector için

```python
from core.ai.tool_impls import build_tool_registry

reg = build_tool_registry()
reg.names()          # ['load_hologram', 'set_recon_param', ...]
reg.schemas()        # OpenAI tools-array (frontend tooltip için description)
spec = reg.get("goto_cell")
spec.name, spec.description, spec.parameters  # JSON-Schema
spec.requires_gui_thread, spec.irreversible
```

Tool inspector UI: chat history'den her tool çağrısını seç → schema'yla
re-render et.

### d) `core.audit.get_audit_log()` — Tool çağrı geçmişi

```python
from core.audit import get_audit_log

log = get_audit_log()
log.entries_today(limit=200)  # list[dict] — her satır JSONL row
# Her entry: {timestamp, action, params, result_summary, user, operator, …}
# Tool çağrıları: action="ai.tool.<name>.start" / ".done"
```

Tool replay UI: `ai.tool.*` filtreli listing → her birini tıklayınca
`spec.parameters`'a göre form render et.

### e) `core.ai.context.StateSnapshot` — Status displays

```python
from core.ai.context import StateSnapshot
snap = main_window._ai_panel._build_snapshot()
# loaded_path, loaded_shape, loaded_dtype
# recon_params, autofocus_params, qpi_params (current sidebar values)
# last_recon_summary, last_af_summary, last_qpi_summary, last_depth_summary
# stage_position_mm, audit_tail
```

### f) `core.settings_schema.AIDefaults` — Settings persist

```python
from core.settings_schema import AIDefaults
from gui.settings_store import load, save

s = load()        # AppSettings
s.ai.enabled, s.ai.endpoint_url, s.ai.model_name
s.ai.temperature, s.ai.max_iterations, s.ai.restrict_to_home

# Update:
s2 = s.with_ai(model_name="qwen2.5:7b-instruct")
save(s2)
```

---

## 5. Mevcut GUI bileşenleri

### `AIPanel(QDockWidget)` — sağ dock, chat
- `objectName="ai_panel_dock"`
- Default gizli, `Cmd+Shift+A` veya `ai.toggle_panel` ile aç/kapa
- Children'lar (objectName ile bulunabilir):
  - `ai_model_label` — model adı QLabel
  - `ai_health_label` — `● Connected` / `● Offline`
  - `ai_chat_browser` — QTextBrowser (chat history)
  - `ai_input_box` — QPlainTextEdit
  - `ai_send_button` — QPushButton
  - `ai_stop_button` — QPushButton
- Public methods (frontend bunları çağırabilir):
  - `set_loaded(path, array)` — main_window zaten çağırıyor
  - `set_recon_summary(d)`, `set_af_summary(d)`, `set_qpi_summary(d)`, `set_depth_summary(d)`
  - `apply_ai_settings(settings: AIDefaults)`
  - `stage()` — StageInterface
  - `_sample_map` — SampleMap instance (private ama erişilebilir)
- Signals:
  - `settings_changed = Signal(AIDefaults)` — settings dialog accept'inde

### `AISettingsDialog(QDialog)` — modal ayarlar
- 10 alan (endpoint, model, temp, tokens, iter, timeout, 3 toggle)
- `dlg.exec() == QDialog.Accepted` → `dlg.result_settings()` → `AIDefaults`
- Reusable: yeni bir "advanced AI" dialog yazmak gerekiyorsa bunu compose et

### `MainWindow` integration
- `mw._ai_panel` — AIPanel referansı (None olabilir, AI panel kurulamamışsa)
- `mw._toggle_ai_panel()` — show/hide
- `mw._push_ai_state_*` — recon/af/qpi/depth completion'larında çağrılıyor
- `mw._install_ai_panel()` — `_init_ui` sonunda çağrılıyor

---

## 6. Settings + persistence yolları

| Veri | Yer |
|---|---|
| AI settings (endpoint, model, …) | QSettings INI: `<DHM>/Reconstruction.ini` altında `v2/ai/*` key'leri |
| Sample map | `<root>/users/<sanitised>/sample_maps/<sample_id>.json` |
| Audit log (tool calls dahil) | `~/.dhm-reconstruction/audit/<YYYY-MM-DD>.jsonl` |
| Per-user state | `<root>/users/<sanitised>/` — `core.user_profile.user_state_dir()` |
| Training data (manuel emit) | `data/ai/training_examples.jsonl` (script ile üretilir) |

`<root>` default `~/.dhm-reconstruction`. `user_profile.set_root_dir()` ile
testte / portable kurulumda yeniden yönlendirilebilir.

---

## 7. Komut palette entegrasyonu (`gui/commands.py`)

Yeni AI-related komut eklemek için pattern (`gui/commands_install.py`'a ekle):

```python
reg.register(Command(
    id="ai.show_map",
    title="Show sample map overlay",
    category=Categories.HELP,    # veya yeni Categories.AI ekle
    shortcut="Ctrl+Shift+M",
    hint="Open the cell-map visualization",
    callback=lambda: window._toggle_sample_map_panel(),
    when=lambda: getattr(window, "_ai_panel", None) is not None,
))
```

Sonra `MAIN_WINDOW_COMMAND_IDS` tuple'ına `"ai.show_map"` ekle.

---

## 8. Threading model — frontend'in dikkat etmesi gerekenler

| Thread | İçinde olan |
|---|---|
| **GUI thread** (Qt main) | `MainWindow`, `AIPanel`, sidebar, image grid, tüm widget'lar, `MainWindow._on_*_completed` slot'ları |
| **AIWorker thread** (QThread) | Tek bir `Agent.run()` döngüsü — chat, tool dispatch, HTTP |
| **Pipeline workers** (her biri kendi QThread) | `ReconstructionWorker`, `AutofocusWorker`, `QPIWorker` |

**Frontend kuralları**:
- AI thread'den **asla** QWidget'a dokunma — `QTimer.singleShot` veya
  `QMetaObject.invokeMethod` ile GUI'ye marshal et.
- `SampleMap` ve `MockStage` thread-safe (Lock'lu).
- AI thread'in pipeline tetiklemesi `_wait_for_signal` ile blok'lu — bu
  GUI'yi kilitlemez (QEventLoop event drenajına devam eder).
- Frontend yeni paneller için `Stop` butonu ekleyecekse `worker.requestInterruption()`
  pattern'ı kullansın (AIWorker'ın yaptığı gibi).

---

## 9. Frontend'in yapacağı işler (sırayla, önerilen)

### W1 — Sample map görselleştirme paneli (öncelik 1)
- **Hedef**: `SampleMap` JSON'unu açıp 2-D scatter göster
- **Plug-in noktası**: yeni `gui/panels/sample_map_panel.py` (QDockWidget)
  - `pyqtgraph.PlotWidget` + scatter (cells, x→`stage_x_mm`, y→`stage_y_mm`)
  - Color: `dry_mass_pg` (LUT)
  - Click → emit `cell_clicked = Signal(int)` → main_window'a bağla,
    main_window AIPanel üzerinden `goto_cell` tool'unu çağırsın (veya direkt
    `ai_panel.stage().move_absolute(cell.x, cell.y, cell.z)`)
- **Reusable kaynak**: `gui/panels/phase_panel.py` — pyqtgraph overlay pattern
- **State**: `QFileSystemWatcher` ile sample_map.json'u izle
- **Komut**: `ai.show_map_panel`

### W2 — Device control panel ✅ SHIPPED (v2.2.0-devices, 2026-04-29)
- **Hedef**: shutter/LED/manuel stage move butonları + slider
- **Ship'lenen**: `src/ui2/device_panel.py` (DPG track) — W5 ile birleşik tek panel.
  - **Stage card**: XYZ HUD + jog grid (X±/Y±/Z± + 4 diagonal) + step
    selector (1 µm / 10 µm / 100 µm / 1 mm / 5 mm) + Home + collapsible
    absolute move.
  - **Shutter card**: tek toggle button (Open/Close) + status pill.
  - **LED card**: On/Off toggle + 0–100 % slider + intensity readout.
  - **Connection footer**: stage / shutter / LED bound durumu.
- **Backend bağlantısı**: `core.devices` Protocols (`StageDevice`,
  `ShutterDevice`, `LEDDevice`); pre-wired ``app._stage`` / ``_shutter``
  / ``_led`` varsa onlar, yoksa `make_device("mock_*")` lazy.
- **Threading**: 5 Hz polling timer → `("device_state", snapshot)`
  mailbox event → app shell `_handle_device_state` → `device_panel.
  handle_mailbox_event` → DPG widget refresh. AI thread'in pipeline
  pattern'ı ile bire bir aynı handshake.
- **Komut**: Tools menu → "Devices…" (ayrı shortcut yok; AI panel'in
  yanına eklendi).
- **Test**: `tests/test_ui2_device_panel.py` — 25 test (snapshot, jog,
  home, absolute move, shutter/LED toggle, intensity clamp, step-label
  round-trip, mailbox handler, polling lifecycle).

### W3 — Tool-call inspector (öncelik 3)
- **Hedef**: AI'ın geçmişte ne yaptığını listeleme + replay
- **Veri kaynağı**: audit log (`ai.tool.<name>.{start,done}` rows)
- **UI**: liste sol, seçili row için JSON detay sağ + "Replay" butonu
- **Replay**: aynı tool'u aynı argümanlarla yeniden dispatch et

### W4 — Time-lapse player (öncelik 4)
- **Hedef**: `record_timelapse` çıktısındaki frame summary'lerini grafiklendirme
- **Veri**: `record_timelapse` tool çıktısı `frames` list'i
- **UI**: scrubber + line chart (phase_std / dry_mass / cell_count zaman serisi)
- **Bonus**: her frame için snapshot rebuild (recon at frame's z)

### W5 — Stage position HUD ✅ SHIPPED (v2.2.0-devices, 2026-04-29)
- **Ship'lenen**: W2 ile aynı panelde (Stage card en üstte). Üç satır
  big-mono XYZ readout — `+0.1234 mm` formatı, sabit genişlik (jog
  sırasında rakam atlamaz). 5 Hz polling, listener-bağımsız (Devices
  Protocol `position_um` property'si üzerinden, mm'ye boundary'de
  çevirilir).
- **Status bar varyantı**: bu ship'te dahil değil — panel açıkken zaten
  daha okunabilir bir HUD var. Ayrı status-bar widget'i ileride status
  bar refactor'ında.

---

## 10. Test koşumu (frontend'in CI'ı için)

```bash
# AI testleri (hızlı, ~2 sn)
QT_QPA_PLATFORM=offscreen pytest tests/test_ai_*.py tests/test_stage.py \
    tests/test_sample_map.py tests/test_settings_schema_ai.py

# Frontend yeni panel eklediğinde:
pytest tests/test_<new_panel>.py

# Sıfır regresyon kontrolü:
python scripts/check_bugs.py
```

**Manual smoke** (frontend yerel test için):
```bash
ollama serve &
ollama pull qwen2.5:7b-instruct
python run_app.py
# Cmd+Shift+A → AI panel
# "Sample'ı 0-2 mm grid haritala" yaz → çalıştığını gör
# data/ai/sample_maps/<id>.json oluştuğunu doğrula
# (W1 panel'i bu dosyayı tüketecek)
```

---

## 11. Açık rezervler / bilinen sınırlar

- **Streaming yok** — `requests` sync. Streaming v2'ye ertelendi (`httpx`).
- **Çoklu-turn memory persistence yok** — chat history app restart'ta sıfırlanıyor.
- **Cellpose yok** — segmentation şu an `core.qpi.segment_cell_phase` (threshold).
  v3.0 roadmap'inde Cellpose entegrasyonu var.
- **`AIPanel` workflow-mode bağımsız** — sidebar tab'ları gibi filtrelenmiyor.
- **Tool registry mutate edilemiyor** — runtime'da yeni tool eklenmiyor;
  `build_tool_registry()` üzerinden compile-time. Frontend yeni tool
  eklemek isterse `tool_impls.py`'a yazsın.
- **`acquire_grid` mock** — gerçek camera yok, `capture_frame` loaded hologram'u
  döndürüyor. Hardware geldiğinde swap.

---

## 12. Hızlı başlangıç — yeni panel ekleme

1. `src/gui/panels/<my>_panel.py` yarat:
   ```python
   from PySide6.QtWidgets import QDockWidget, QVBoxLayout, QWidget
   from PySide6.QtCore import Qt, Signal

   class MyPanel(QDockWidget):
       def __init__(self, main_window, parent=None):
           super().__init__("My Panel", parent)
           self.setObjectName("my_panel_dock")
           self._mw = main_window
           # ... build UI ...
   ```
2. `main_window.py:_install_my_panel()` ekle (AIPanel pattern'ı):
   ```python
   self._my_panel = MyPanel(self, parent=self)
   self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._my_panel)
   self.tabifyDockWidget(self._ai_panel, self._my_panel)
   self._my_panel.hide()
   ```
3. `commands_install.py`'a komut ekle (`ai.show_my_panel`).
4. `tests/test_my_panel.py` yaz (`AIPanel` testi şablon).
5. PR'a `docs/AI_HANDOFF.md` güncellemesi de ekle.

---

## İlgili devir dökümanları

- [docs/AI_FINETUNE.md](AI_FINETUNE.md) — Ollama Modelfile + LoRA pipeline
- [docs/AI_FINETUNE_DATA.md](AI_FINETUNE_DATA.md) — Eğitim verisi tasarımı
- [docs/ROADMAP.md](ROADMAP.md) — Genel sürüm planı
- [tasks/lessons.md](../tasks/lessons.md) — Geçmiş hatalar + kurallar

Sorular için: bu dökümanı yazan agent (Claude) hazır, bir sonraki AI-domain
sprintinde devamı geliyor.

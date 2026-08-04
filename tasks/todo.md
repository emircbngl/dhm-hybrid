# Plan — 2026-04-21 — Faz 1.1 (v1.0.1-ux)

## Context
Faz 1 (v1.0.0, 2026-04-20) teknik review'ı geçti ama lab pilotunda **hem teknik hem görsel** UX problemiyle geri döndü (Anna / Lindqvist lab). Dil karışımı, sessiz recon/QPI, transient hata, parameter amnezisi, klavye yokluğu, cancel yokluğu — altı şikâyet. Analiz: bunların yarısı yüzey (copy, tema, klavye) ama diğer yarısı **worker-level mimari eksiklik** (progress signal yok, error shape lossy, interruption check yok, command registry yok, settings schema typed değil).

Hedef: Faz 2'ye (AI segmentasyon, licensing, PDF) geçmeden v1.0.1-ux patch'iyle dön. Rams/Ma/Ive penceresinden: *"Dış sessizlik, iç rijitliğe borçlu."*

## Strateji: iki tier
- **Tier 0 — invisible plumbing (~8 gün):** Faz 1.1'in %70'i. Lab açsa hiçbir görsel fark görmez. Ama üstüne güvenilir Tier 1 çakılır.
- **Tier 1 — visible design (~4 gün):** Tier 0'ı *tüketir*. Command palette, toast, sessiz progress line, persistence, dil reduction.

---

## Tier 0 — plumbing ✅ SHIPPED (2026-04-21)

### T0.1 · `src/core/progress.py` — progress protocol ✅
- [x] `ProgressEvent(operation_id, operation_kind, phase, phase_index, phase_count, pct, elapsed_s, eta_s, message, done, cancelled)` frozen dataclass
- [x] `Operation` context manager — elapsed tracking, ETA hesaplama, phase transitions
- [x] Pure Python; Qt bağlantısı worker katmanında
- [x] `tests/test_progress.py` (9 test)

### T0.2 · `src/core/errors.py` — structured error bus ✅
- [x] `Severity` enum (INFO / WARN / ERROR / FATAL)
- [x] `ErrorEvent(severity, title, cause, action, context, trace, timestamp, event_id)` frozen dataclass
- [x] `ErrorCenter` singleton: `subscribe(fn, min_severity=)`, `emit(event)`, bounded session history
- [x] Built-in sinks: `logger_sink` + `audit_sink`
- [x] `tests/test_errors.py` (14 test)

### T0.3 · `src/core/settings_schema.py` + `src/gui/settings_store.py` ✅
- [x] `SCHEMA_VERSION = 2`
- [x] `ReconDefaults`, `AutofocusDefaults`, `QPIDefaults`, `IODefaults`, `AppSettings` dataclass'ları
- [x] `gui/settings_store.py` — `QSettings` IniFormat + UserScope; load/save; v1 → v2 migrator
- [x] `validate()` tipli uyarılarla; invalid on-disk state → defaults fallback
- [x] `tests/test_settings_schema.py` (13 test)

### T0.4 · `reconstruction_worker.py` refactor ✅
- [x] `_process` → 7 faz (preprocess, offaxis, freqfilter, propagate, refsub, finitecheck, unwrap)
- [x] Her faz sınırında `_check_cancel()` → `OperationCancelled`
- [x] `progress = Signal(object)` — `ProgressEvent` emit
- [x] `error_event = Signal(object)` — structured; legacy `error_occurred = Signal(str)` yan yana korunur
- [x] `_emit_error(exc, job)` faz bağlamı + Rams-#4 action hint
- [x] `tests/test_reconstruction_worker.py` (7 test)

### T0.5 · `qpi_worker.py` + `autofocus_worker.py` ✅
- [x] QPI: 4 faz (validate, refsub, bgcorrect, compute) + `error_event`/`progress` signals
- [x] AF: minimal surgery — `error_event = Signal(object)` eklendi, mevcut cooperative cancellation korundu
- [x] Her üç worker'da da `requestInterruption()` semantiği
- [x] `tests/test_qpi_worker.py` (10 test) + `tests/test_autofocus_worker.py` (6 test)

### T0.6 · `src/gui/commands.py` — command registry ✅
- [x] `Command(id, title, category, callback, shortcut, hint, when, visible_in_palette)` frozen dataclass
- [x] `CommandRegistry` — thread-safe, ordered, search + by_category + invoke + when-predicate
- [x] `gui/commands_install.py` — 9 main-window commands kayıt + QShortcut üreteci
- [x] `main_window._setup_shortcuts` + `_init_menus` registry'den besleniyor
- [x] `tests/test_commands.py` (23 test)

### T0 Verification ✅
- [x] `pytest tests/` — 107 test yeşil (1.23s)
- [x] Esc cancels: reconstruction + QPI + AF — cancellation testleri mevcut
- [x] ErrorEvent phase context: z_mm/wavelength_nm/pixel_um/n_sample/n_medium test assert'leri
- [x] v1 → v2 migration: `test_v1_to_v2_migration_stamps_version` yeşil, mevcut `window/geometry` korunuyor
- [ ] Manuel GUI smoke (Esc kesiyor, toast gösterisi) — Tier 1 UI landingi sonrası kullanıcı oturumunda

---

## Tier 1 — visible design

### T1.1 · `src/gui/widgets/command_palette.py` ✅
- [x] `⌘K` açar; fuzzy search; Enter çalıştırır
- [x] `CommandRegistry`'yi tüketir; dinamik `when` değerlendirmesi (greys out disabled komutlar)
- [x] `tests/test_command_palette.py` (5 test, offscreen Qt)

### T1.2 · `src/gui/widgets/toast.py` + `src/gui/widgets/error_drawer.py` ✅
- [x] Toast: sağ üst overlay, dismiss'e kadar kalır, max 3 stack
- [x] Başlık + cause + action + "Show log ›" chevron
- [x] Drawer: `QDockWidget`, session error history, timestamped, traceback collapsible
- [x] Her ikisi de `ErrorCenter` subscriber'ı; severity floor = WARN
- [x] `tests/test_toast_and_drawer.py` (8 test)

### T1.3 · `src/gui/widgets/progress_line.py` ✅
- [x] Status bar'da 1px çizgi; `ProgressEvent`'i tüketir
- [x] <500ms gizli, 500ms–5s line-only, >5s line + caption + Esc ipucu
- [x] `cancel_requested` signal → `_cancel_active_worker` (QPI → AF → recon)
- [x] `tests/test_progress_line.py` (8 test)

### T1.4 · Parameter persistence wiring ✅
- [x] `src/gui/persistence.py` — `apply_settings` / `collect_settings` tabs ↔ `AppSettings`
- [x] `_load_persisted_settings()` `__init__` sonunda; sidebar widgets'a inject
- [x] `_persist_current_settings()` recon/AF/QPI başarılı sonrasında
- [x] `_update_io_history()` altı `QFileDialog` site'ında: load hologram (toolbar), reference hologram, video output dir, snapshot, export view, export panel, report save
- [x] `last_folder` toolbar default_dir'e; `last_report_folder` rapor dialog default'una
- [x] `tests/test_persistence_wiring.py` (9 test) + manuel smoke: fresh QSettings → save → reload round-trip OK, MainWindow boot + apply OK

### T1.5 · Dil reduction + CI guard ✅
- [x] 12 Türkçe string çevrildi: AF fallback mesajı, amp+phase panel menüleri, export dialog, report dialog, 4 status mesajı
- [x] Verbosity azaltıldı: "Line Profile başlat" → "Line profile"; "3D Yüzey göster" → "3D surface"; "Görüntüyü dışa aktar…" → "Export image…"
- [x] `scripts/check_language.py` — standalone Python scanner; pre-commit olmadan da manuel çalıştırılabilir
- [x] `.pre-commit-config.yaml` — `local` hook; Türkçe karakter görürse commit'i bloklar
- [x] Smoke: clean pass (exit 0), dirty detect (exit 1, satır numarası + uyarı)
- [x] `src/gui` grep sonrası sıfır Türkçe karakter (string + comment)

### T1.6 · Esc = cancel everywhere ✅
- [x] `_setup_shortcuts`'ta `QShortcut(Qt.Key.Key_Escape, self)` → `_cancel_active_worker`
- [x] WindowShortcut context — modal dialog'lar ve error drawer kendi Esc'lerini korur
- [x] Walker status bar'a "QPI/Autofocus/Reconstruction: cancel requested" feedback yazar (4s)
- [x] Hiç worker çalışmıyorsa Esc sessiz — kullanıcıyı rahatsız etmez
- [x] Progress line'ın "Press Esc to cancel" caption'ı zaten T1.3'te şiplenmişti; Esc artık o vaadi yerine getiriyor
- [x] `tests/test_cancel_walker.py` (6 test): priority order, stopped-skip, stop-vs-cancel, exception swallowing, Esc wiring smoke
- [x] Autofocus'un kendi cancel button'u korundu (mouse'u tercih edenler için) — Rams #4 "understandable"

### T1.7 · Context-sensitive sidebar (stretch)
- [ ] Recon/Focus/QPI tab'leri → mod switch (aynı sütun farklı içerik)
- [ ] `CommandRegistry.enabled_when`'e bağlı göster/gizle
- [ ] Stretch çünkü büyük görsel refactor; T1.1–T1.6 bittikten sonra yapılırsa yapılsın

---

## Verification

- [ ] `pytest tests/` — hepsi yeşil
- [ ] `bench_af.py --ref-only` — hâlâ ≈42–45mm
- [ ] Smoke scenario: uygulamayı aç → parametrelerin hatırlandığını gör → Cmd+K → "recon" → Enter → sessiz progress çizgisi → Esc → cancel olduğunu gör → yanlış param gir → toast kart → "Show log ›" → drawer
- [ ] Dil guard: CI'da Türkçe string commit'i red etsin
- [ ] Audit log'da Tier 0 yeni events: `operation_started`, `operation_completed`, `error_event`

## Review

v1.0.1-ux shipped 2026-04-21. T0 + T1.1–T1.6 tamamlandı, T1.7 (context sidebar) v1.4 UI Redesign sprint'inde ele alındı ve shipped.

---

# Plan — 2026-04-24 — v2.0.2-polish (Dear PyGui frontend)

## Context
Pilot test sonucu (Emir, 2026-04-24): v2 (Dear PyGui) UI'ında 15 bulgu. 2 blocker + 5 major + 8 minor. Kapsam kararı **A-iii** (hepsi paketlensin, "v2.0.2-polish"), fallback kararı **B** (macOS'ta drop-zone affordance + click fallback, PyObjC yok).

Üç somut CEO şikayeti (drop fail, window sizing, persistence yok) planın omurgası; kalan 12 bulgu organik olarak bu omurgaya iliştirilmiş — persistence yazarken file-dialog default_path bedava, layout refactor'u sırasında info_text birikim hijyeni ucuza gelir.

QA-scout bulguları ve Plan-architect mimari kararları referans; çalışma dosyaları:
- `src/ui2/app.py` — ana shell (1504 satır)
- `src/ui2/image_panel.py` — ZoomableImagePanel
- `src/ui2/widgets.py` — CommandPalette, ToastStack, PresetChips helpers
- `src/ui2/surface.py` — subprocess 3D
- `src/core/settings_schema.py` — typed AppSettings (paylaşılan, Qt-free)

---

## T1 — Blockers

### T1.1 · Drag-and-drop doğrudan desteklenmiyorsa görünür drop zone
- [ ] `app.py` init: `self._drop_supported = sys.platform != "darwin"` + `DHM_FORCE_DROP` env override
- [ ] `_build_ui` panel grid üstüne 48px yüksekliğinde `drop_zone` child_window, label capability-aware ("Drop hologram here" / "Click to load hologram")
- [ ] `item_handler_registry` → click = `_on_load_clicked`
- [ ] Status bar default message capability-aware (`Ready. Drag…` / `Ready. Click the drop zone…`)
- [ ] `set_viewport_drop_callback` yalnızca supported ise bind, try/except sessiz fail değil — explicit log + toast
- [ ] Test: `tests/test_ui2_dnd.py` — (a) drop_zone label capability'ye göre, (b) click handler `_on_load_clicked` tetikliyor, (c) payload parsing list[str]/list[dict]/str için path çıkarıyor, (d) non-image uzantı warn üretiyor

### T1.2 · Viewport sizing — akıllı default + responsive PREVIEW_SIZE
- [ ] `DEFAULT_SIZE` class sabitini kaldır, `_compute_initial_viewport()` staticmethod ekle (`tkinter.Tk().winfo_screenwidth()` prob, immediate destroy)
- [ ] `PREVIEW_SIZE` class sabitinden `self._preview_size` instance attr'a
- [ ] Viewport boyutu: `max(1100, min(sw*0.85, 1600)) × max(720, min(sh*0.85, 1000))`
- [ ] Tier: viewport_w ≥ 1500 → 512, ≥ 1250 → 384, else 288
- [ ] `ZoomableImagePanel.resize(new_size)` yeni metod — texture re-register + plot limits reset
- [ ] `set_viewport_resize_callback` → tier boundary'de `resize()` çağır, hysteresis (aynı tier'da idempotent)
- [ ] `no_scrollbar=True` kaldır `content_row`'dan (QA #2 ikincil sebep) — dokunulmaz yerde scroll sigortası
- [ ] Test: `tests/test_ui2_layout.py` — (a) 3 monitör tier'ı doğru, (b) `sidebar(320) + 2*(preview+16) ≤ viewport_w` invariant, (c) resize tier geçişinde tam bir kez tetiklenir

---

## T2 — Major fixes

### T2.1 · Persistence — `ui2_state.json` + paylaşılan `AppSettings`
- [ ] `src/core/settings_schema.py`: `Ui2State` dataclass ekle (viewport_w/h, theme, sample_id, workflow_mode, selected_preset, recent[str], last_dir, reference_path, subtract_reference, params)
- [ ] `AppSettings`'e `ui2: Ui2State` field ekle, `SCHEMA_VERSION = 3`
- [ ] `_v2_to_v3(raw)` migrator — eksik ui2 alanı default'la hydrate
- [ ] `src/ui2/state_store.py` yeni dosya: `load()` / `save()` — `~/.dhm-reconstruction/ui2_state.json` atomic (tmpfile + `os.replace`)
- [ ] `app.py.run()` başında `load()` → `_params`, `_theme_name`, `_recent`, UI widget'lar hydrate
- [ ] Debounced save: `_mark_dirty()` helper, render loop'ta `_dirty + timer` kontrolü, `threading.Thread(target=save, daemon=True)` — UI thread bloklamaz
- [ ] `finally` block'ta senkron son save
- [ ] Hydrate: `_on_param_changed`, `_apply_theme`, `_apply_preset`, `_on_workflow_changed`, `sample_id_input`, recent menu → `_mark_dirty()`
- [ ] `_last_save_error` + Help menüde "Show state path" item (debug)
- [ ] Test: `tests/test_ui2_state.py` — roundtrip, missing file, old v1 JSON (ui2 yok), corrupt JSON → default fallback, atomic (tmp leftover sağlam), debounce (10 rapid change = 1 save)

### T2.2 · File dialog default_path persist (T2.1 bedava yan ürünü)
- [ ] `file_dialog_open` oluşturulduğunda `default_path=self._last_dir` (state'ten gelir)
- [ ] `_on_file_selected` / drop / recent load → `self._last_dir = path.parent`, `_mark_dirty()`
- [ ] Reference dialog da aynı mekanizma

### T2.3 · Info text biriktirme bugfix
- [ ] `_handle_qpi`, `_handle_depth`, `_apply_result`'da info text'i **append değil replace** — QPI/depth sonuçları için ayrı section içinde render et, recon değişirse reset
- [ ] `_clear_depth_overlay` → info text'ten depth bloğunu kaldır (veya base recon info'ya döndür)
- [ ] Yardımcı: `_compose_info_text(recon=…, qpi=…, depth=…, ref=…)` — tek kaynak
- [ ] Test: `tests/test_ui2_info_compose.py` — QPI+depth ekleyip clear'dan sonra base state

### T2.4 · `btn_surface` race — phase texture başarısız olduğunda disabled kalsın
- [ ] `_apply_result` başında `btn_surface` disabled bırak, sadece texture push başarılı olduktan **sonra** enable et
- [ ] Exception durumunda ayrıca disabled garanti
- [ ] `_on_open_surface` zaten `_last_recon is None` kontrolü yapıyor; ek olarak `phase.size > 0` guard

### T2.5 · Reference path info panelinde görünür
- [ ] `_compose_info_text` (T2.3) içinde "Reference: <path.name or '(none)'>" satırı
- [ ] Subtract aktif mi (`subtract_reference` bool) o satırda renkle ayrıştır

---

## T3 — Minor / Polish

### T3.1 · Workflow combo gerçek iş yapsın (sidebar section hide/show)
- [ ] Sidebar section'ları tag'le (`section_sample`, `section_preset`, `section_params`, `section_reference`)
- [ ] Mode → visible section seti tablosu:
  - Reconstruct: sample, preset, params, reference
  - Analyse: sample, params (readonly highlight), QPI toolbar prominent
  - Report: sample, reference summary, export butonları prominent
- [ ] `_on_workflow_changed` → `dpg.configure_item(tag, show=…)`
- [ ] State'e persiste edilir (T2.1)
- [ ] Test: `tests/test_ui2_workflow.py` — her mode için görünür section seti

### T3.2 · `btn_reconstruct` disabled reason görünür
- [ ] Disabled iken tooltip: "Load a hologram first (File → Load…)"
- [ ] Enabled olunca tooltip: "Run reconstruction (Ctrl+R / ⌘R)"

### T3.3 · Mac'te shortcut label'ları temizle
- [ ] `sys.platform == "darwin"` → tüm label'larda sadece "⌘O", diğer platformlarda "Ctrl+O"
- [ ] Shortcuts modal genişliğini 360→420 yükselt

### T3.4 · Preset chip seçim state görünür
- [ ] `widgets.PresetChips` helper zaten mevcut — direkt entegre et, manuel button listesi yerine
- [ ] `set_active(name)` tıklamada + state hydrate'inde çağrılır

### T3.5 · Onboarding flag reset path (Help menüde)
- [ ] Help → "Reset first-run onboarding" item — flag dosyasını sil + toast
- [ ] `_show_onboarding_manual` zaten var, sadece "bir daha otomatik göstermesin" checkbox modal içine

### T3.6 · Depth map semantik doğru colormap
- [ ] `_to_rgba` `colormap="depth"` ekle — linear viridis-benzeri ramp (periodik değil)
- [ ] `_handle_depth` → `"phase"` yerine `"depth"` geç

### T3.7 · Command palette boş match + Enter degrade
- [ ] `CommandPalette.invoke_top`: filtered boşsa toast "No matches", palette açık kalır (Esc ile kapanır)
- [ ] Ya da: açılır "Nothing matches 'xyz' — Esc to close" hint

### T3.8 · Surface subprocess tmp cleanup loud fail
- [ ] `surface.py`'de `tempfile.TemporaryDirectory()` context manager kullan → leak garantisi OS'a bırak
- [ ] Cleanup exception'ını warn-log et (silent yutma yok)

---

## T4 — Verification

- [ ] `python -m pytest tests/test_ui2_*` → tüm yeni testler yeşil
- [ ] `python -m pytest` → full suite hâlâ yeşil (v1 regresyon yok — core paylaşımlı schema bump migration test'leri)
- [ ] `python3 scripts/check_language.py` → `src/ui2/` dahil clean
- [ ] Manuel smoke: uygulamayı aç → viewport düzgün sığıyor → macOS'ta drop zone görülüyor, click ile dosya yükleniyor → parametre değiştir → kapat/yeniden aç → parametre hatırlandı → recent menü dolu → tema hatırlandı → workflow mode değişince sidebar sections update
- [ ] macOS küçük ekran (1366×768) ve büyük ekran (2560×1440) viewport tier testi
- [ ] v1 ile paralel çalıştır — v1 parametreleri v2'yi bozmamalı (iki farklı serializer aynı schema)

## Sıra
T1.2 (sizing) → T1.1 (drop) → T2.1 (persistence, omurga) → T2.2/2.3/2.4/2.5 (persistence'a tak) → T3.1 (workflow) → T3.2–3.8 (minor batch) → T4 (verification).

Tahmini süre: ~1 hafta (5-6 iş günü, testler dahil). Her T bitiminde interim commit yok — pilot snapshot Worktree/ altında plain file copy (kullanıcı kuralı).

## Review

v2.0.2-polish shipped 2026-04-24. QA-scout'ın 15 bulgusu + 3 plan-architect omurga çözümü tek sprint'te tamamlandı.

### Shipped
- **T1.2 (blocker)** — `_compute_initial_viewport()` + tkinter screen probe + tier'lı `PREVIEW_SIZE` (288/384/512, viewport genişliğine göre); `ZoomableImagePanel.resize()` eklendi, `set_viewport_resize_callback` live relayout için bağlandı; `content_row` üzerindeki `no_scrollbar=True` kaldırıldı (sigorta açık).
- **T1.1 (blocker)** — macOS'ta `set_viewport_drop_callback` Cocoa binding'i sessiz olduğu için platform-aware capability detect + görünür `drop_zone` child_window + click-to-browse fallback; status bar + onboarding metinleri dürüst kapasite mesajı veriyor; `DHM_FORCE_DROP=1` env override gelecekte upstream fix için geri dönebilme yolu.
- **T2.1 (major)** — `Ui2State` dataclass `src/core/settings_schema.py`'a eklendi, `SCHEMA_VERSION` 2→3 bumpı, hem v1 (QSettings) hem v2 (JSON) tarafına v2→v3 migration; yeni `src/ui2/state_store.py` (atomic write + debounced async saver + migration zinciri v1→v2→v3); `DhmApp.__init__` load, `run()` hydrate, `_mark_dirty()` her anlamlı callback'te, `finally`'da `flush_now()`.
- **T2.2 (major)** — `file_dialog_open` ve `file_dialog_reference` artık `_last_dir`'den açılıyor; her load path `_last_dir`'i güncelliyor, `_mark_dirty()` tetikliyor; Recent menü File → Recent submenu'den hydrate ediliyor.
- **T2.3 (major)** — `_compose_info_text()` helper tek kaynak (recon, QPI, depth, reference stateinden); append yerine replace; `_clear_depth_overlay` info text'i yeniden render ediyor.
- **T2.4 (major)** — `_apply_result`'ta `btn_surface` başta disabled, texture push başarılı olduktan **sonra** enable oluyor + `phase.size > 0` guard. Error path'te de disabled garantisi.
- **T2.5 (major)** — Reference path + subtract durumu info panelde `Reference: ref.tif (on)` / `(loaded)` / `(none)` olarak görünür.
- **T3.1 (minor)** — Workflow combo gerçek iş yapıyor: `_WORKFLOW_SECTIONS` tablosu her mode için visible section set'i verir, `_apply_workflow_visibility` `dpg.configure_item(tag, show=…)` ile uygular (Reconstruct/Analyse/Report üçü farklı sidebar'lar).
- **T3.2 (minor)** — `btn_reconstruct_tip` tooltip: disabled iken "Load a hologram first…", enabled iken `"Run reconstruction (⌘R)"` (Mac) / `(Ctrl+R)`.
- **T3.3 (minor)** — `DhmApp._shortcut("Ctrl+O")` Mac'te "⌘O", diğer platformlarda "Ctrl+O" döner; menü/tooltip/shortcut modal tek kaynaktan türer; shortcut modal 420×260'a genişledi.
- **T3.4 (minor)** — Manual preset button satırı yerine `PresetChips` helper; seçilen chip disabled-looking (aktif işaret), state `_selected_preset`'e yazılır, JSON'a persist edilir.
- **T3.5 (minor)** — Help menüde "Reset first-run onboarding" — flag dosyasını siler + toast + anında welcome wizard'ı açar.
- **T3.6 (minor)** — `_to_rgba(colormap="depth")` yeni; viridis-benzeri monotonik ramp (periyodik değil — z physical meter, phase wheel yanlıştı). `_handle_depth` artık "phase" yerine "depth" kullanıyor.
- **T3.7 (minor)** — `CommandPalette.invoke_top` boş match'te sessizlik yerine "No matches — try a different query or press Esc to close." satırı `WARN` renginde gösterir; Enter acknowledgement görünür.
- **T3.8 (minor)** — `surface.py` tmp cleanup artık `warning:` stderr'e yazar, sessiz yutmuyor — /tmp accumulation biter.

### Tests
37 yeni test:
- `tests/test_ui2_state_store.py` (13) — save/load roundtrip, missing file, corrupt JSON, empty file, atomic (no tmp leftover), v1→v3 migration, v2 ui2-yok migration, forward-compat unknown keys, recent list normalization, DebouncedSaver coalesce + no-dirty-noop + flush_now sync.
- `tests/test_ui2_logic.py` (24) — viewport tier boundaries (7 parametre), tier invariant (sidebar + 2 panel genişliği ≤ viewport_w), Mac/non-Mac shortcut format, workflow sections her mode'u kapsıyor, Report modunda preset + reference gizli, colormap'lerin (gray/depth/phase) range'i + depth monotonikliği + phase periyodikliği, info composer boş/hologram-only/ref-on/ref-loaded/ref-none/recon+depth+clear senaryoları, drop zone label capability-aware + status ready text.

### Verification
- `PYTHONPATH=src:tests ../Phyton/venv/bin/python -m pytest` → **338 passed in 9.51s** (önceki 301'e 37 eklendi, hiçbir regresyon yok).
- `../Phyton/venv/bin/python3 scripts/check_language.py` → **exit=0** (hiçbir Türkçe karakter src/ui2/ veya src/gui/ içinde).
- Smoke: DhmApp() init'i macOS'ta `drop_supported=False`, viewport ~1453×945, tier=384 olarak çıktı verdi (kullanıcının ekranına uygun responsive default).

### Kararlar ve gerekçe notları
- Ui2State'i v1 AppSettings'e eklemek v2'ye ayrı bir schema yazmaktan daha DRY — aynı migration zinciri, iki farklı serializer (QSettings + JSON).
- Drop zone için PyObjC seçilmedi (kullanıcı "fallback OK" dedi); görünür click-target zaten macOS'ta kullanışlı affordance.
- Debounced saver 1.5s delay ile: hızlı parametre değişimlerinde disk yazımını 10 değişikliğe 1 kez getiriyor; test 0.05s delay ile aynı davranışı 1ms seviyesinde doğruluyor.
- `_workflow_mode` → sidebar section toggle: Report modunda preset + reference gizlenir (lab kullanıcısının mental yükünü azaltır), Analyse modunda preset gizlenir ama params görünür (read-only ilerde eklenecek), Reconstruct tam fonksiyon.

### v2.0.3 backlog (yeni görünen)
- Drop zone üstünde reconstruction progress bar / live indicator (şu an status text tek satır — toast yeterli ama minimal).
- Workflow "Report" modunda export butonlarının prominent hale gelmesi (şu an sadece gizleme var, highlight yok).
- Preset save/load (şu an preset'ler dict sabit — kullanıcı kendi preset'ini kaydetmek isteyebilir).

---

# Plan — 2026-04-24 — v2.0.3-physics-gap (v1→v2 port accuracy debt)

## Context
CEO pilot testte "magnification neden sorulmuyor?" dedi. Tek soru 9 bilimsel parametre gap'ini ortaya çıkardı — v2 port audit'i yapmadığımız için v1'in bilimsel sidebar'ı eksik taşınmıştı. 40× mikroskop setup'ında z propagation M² kayıyordu, QPI dry mass yanlış `n`'le hesaplanıyordu, autofocus 1 metriğe kilitliydi. CEO: "böyle hatalar bir daha görmek istemiyorum." Hemen fix — Explore subagent audit çıkarttı, tüm accuracy item'ları aynı sprint'te kapandı.

## Shipped
- **Magnification + pixel_is_effective** — `ReconParams.effective_pixel_um()` helper; sidebar `×M` spinbox + "Pixel is already effective" checkbox; reconstruction + workers (autofocus, QPI, depth) effective pixel'i kullanıyor; preset'ler sane M default'ları taşıyor (Cell ×40, USAF ×10, Film ×1, Custom ×1); TIFF metadata `magnification` auto-detect + toast ile bildirim; info panel "Effective px: X.XXX µm = Y.YYY µm / ×M" göstererek pipeline matematiğini transparanlaştırıyor.
- **QPI n_sample + n_medium** — `ReconParams`'a field olarak eklendi; sidebar'da "QPI" alt bölümünde iki spinbox; `workers.run_qpi` + `run_qpi_batch` explicit arg gelmezse `params.n_sample`/`params.n_medium`'u kullanıyor (backwards-compatible).
- **Autofocus metric combo** — `ReconParams.autofocus_metric: str` + sidebar "Autofocus" bölümünde combo; `available_focus_metrics()` helper `FocusMetric` enum'unu otomatik listeliyor; `_metric_from_params()` string→enum, unknown name'de `LAPLACIAN_VARIANCE`'e fallback; bütün workers (autofocus, multi-focus, QPI batch, depth map) `params.autofocus_metric`'i honor ediyor.
- **TIFF metadata auto-detect** — `_apply_detected_metadata()`; magnification, pixel_size_m, wavelength_m her biri ayrı toast; magnification detect edilirse `pixel_is_effective` otomatik False'a çekiliyor (dosya header'ı camera pixel taşır).
- **Nyquist smell-test** — `_on_param_changed` effective_px > λ/2 olduğunda status bar warn: "double-check ×M or 'pixel is effective'"; blocker değil, eğitim amaçlı.
- **Persistence** — Ui2State'e 5 yeni field (magnification, pixel_is_effective, n_sample, n_medium, autofocus_metric); `SCHEMA_VERSION` 3→4; `_v3_to_v4` migration eski state dump'larını v2.0.2 davranışıyla bit-identik backfill'liyor (M=1, pixel_is_effective=True, n_sample=1.38, n_medium=1.337, autofocus_metric=LAPLACIAN_VARIANCE). Qt tarafı (`gui/settings_store.py`) için `_migrate_v3_to_v4` no-op version stamp.

### Tests (tests/test_ui2_scientific_params.py, 24 yeni)
- `effective_pixel_um` 5 parametreli edge case (M=40 effective=False → 3.45/40, M=40 effective=True → 3.45 pass-through, M=1 no-op, M=0 div-zero guard, vs.)
- Preset sanity: Cell ×40 + pixel_is_effective=False, Film ×1 + effective=True, USAF ×10 + TENENGRAD metric, her preset 10 field'ı da taşıyor.
- State roundtrip: magnification + n_sample + metric disk-to-memory roundtrip kimliği.
- Migration: v3 payload (ui2 yok field'lar) hydrate'de v1-match defaults; v1 payload (ui2 hiç yok) direct v4'e; forward-compat unknown field drop.
- Info composer: Effective px satırı divide edilince "X.XXX = Y.YYY / ×M" göster, pixel_is_effective=True iken division sembolü yok.
- Metric listesi: ≥5 metric, hepsi uppercase; round-trip string→enum; unknown name fallback LAPLACIAN_VARIANCE.
- Metadata auto-detect: magnification/pixel_size/wavelength her biri ayrı, boş meta toast yok, aynı value detect toast yok.

### Verification
- `PYTHONPATH=src:tests ../Phyton/venv/bin/python -m pytest` → **362 passed in 9.35s** (v2.0.2'den 338 + 24 yeni scientific param test = 362, sıfır regresyon).
- `scripts/check_language.py` → exit=0.
- Smoke: macOS'ta `app._viewport_w=1453, _preview=384, _drop_supported=False, params.magnification=1.0, effective_pixel_um()=5.0, QPI n_sample=1.38, AF metric=LAPLACIAN_VARIANCE`.

### Açık kalan (v2.0.4 için)
- `z_min/z_max` autofocus range hardcoded `-25/+25 mm` (v1 user-editable, -1/+1 default). Fast DHM setup'larda gereksiz scan yapıyor.
- `subtract_mean` + `hann_window` `ReconDefaults`'ta var ama v2 pipeline kullanmıyor.
- Phase unwrap method seçimi (şu an sadece GRADIENT_INTEGRATION).
- FFT backend seçimi (PyFFTW/MLX/scipy/numpy — v1 combosu).

Bunlar **accuracy** değil (Nyquist/physics doğru sayılıyor default'lar), `performance` veya `niche-case` kategorisinde — ayrı bir sprint'te. `tasks/lessons.md`'e "v1→v2 port audit" dersi eklendi; bir sonraki port benzer bir yeniden keşfe fırsat vermemeli.

### 2026-04-24 Re-audit (CEO istedi) — kalan 25+ feature gap

İkinci Explore audit'i: v1 `src/gui/widgets/` (14 dosya) + `src/gui/sidebar/` (7 tab) + `main_window.py` full scan. Scientific gap'ler v2.0.3'te kapanmıştı; bu audit **UI/UX + compliance + ergonomi** kategorilerine baktı. Tekrar eden kayıp özellikler (v1'de var, v2'de yok):

**CEO-priority (compliance + science workflow):**
- **Audit logging JSONL** — `core/audit.py` mevcut, Qt tarafı her recon/QPI çağrısında `get_audit_log().write(...)` yapıyor; v2'de hiç `audit` referansı yok. Lab compliance için blocker — her reconstruction'ın trail'i olmalı.
- **Esc-to-cancel active worker** — v1'de `main_window.py:397-403` cancel walker var, her çalışan worker `requestInterruption()`'la durdurulur; v2'de Esc sadece status bar'ı "Ready."'ye çekiyor, iş devam ediyor.
- **Line profile tool** — v1 toolbar + amplitude panel right-click → intensity profile dialog. Bilim workflow için standart tool; v2'de yok.
- **Batch mode dialog** — v1'de `dialogs/batch_render_dialog.py` ile dizin seçip çoklu hologram batch işlenebiliyor; v2'de backend (`core/batch_renderer.py`) var ama UI yok.

**Next-sprint (UX):**
- Maximize panel shortcuts (Ctrl+1/2/3/4 → tek panel fullscreen, Ctrl+0 restore)
- Error drawer (side panel — v1'de `widgets/error_drawer.py`, `ErrorCenter` bus'a subscribe); v2 toast'tan ibaret
- Progress line/spinner (v1 `widgets/progress_line.py`, Esc hint'li); v2 sadece status text
- Right-click context menus (phase panel → 3D surface, amplitude panel → line profile)
- Validation dots (widgets/validation_dot.py — inline param bound check); v2'de helper hazır ama hiç bağlı değil
- Profile combo (setup + camera profile toolbar; v1'de çoklu lab setup switch)
- Crop tool (ROI overlay)
- Keyboard shortcuts dialog daha geniş; v2'de 6 satır, v1'de tam liste

**Backlog:**
- Camera live feed + record (v1 `camera_tab.py` + `workers/acquisition_worker.py`)
- Record tab (hologram video recording)
- Crash handler (sys.excepthook) — v1'de `core/crash_handler.py`, v2'de wire edilmiyor
- High-contrast theme'in WCAG-AA compliance'ı v1'de test ediliyor, v2'de palette var ama accessibility test yok
- F5/refresh shortcut

**CEO'ya takip soru (kullanıcı kendi belirtti):**
- **Multi-focus arama doğru çalışıyor mu?** `find_focus_candidates` core'da test ediliyor (`tests/test_focus_candidates_dialog.py`), ama v2 UI üzerinden z-range / metric / prominence ayarları doğru aktarılıyor mu scan'e — accuracy doğrulama gerekli. Ayrı ticket, v2.0.4-multifocus-validation.

### Bir sonraki sprint kararı (CEO onayı ile)
Scientific gap'lar (magnification + QPI params + AF metric) **v2.0.3'te kapandı**. Yeni bulunan gap'ler bilimsel doğruluk değil, UX/compliance. CEO isterse:
- **v2.0.4-compliance**: audit log + Esc-to-cancel (2 CEO-priority, fiziksel değil ama lab için blocker)
- **v2.0.5-workflow-tools**: line profile + batch dialog + maximize panels + error drawer
- **Multi-focus validation**: kullanıcının takip sorusu — `find_focus_candidates` v2 üzerinden ihtiyaç duyulan accuracy'de mi, sentetik test ekle.

---

# Plan — 2026-04-24 — v2.0.4 + v2.0.5 mega sprint (CEO "hepsi")

## Context
CEO "hepsi" dedi. Re-audit'in CEO-priority + next-sprint + backlog item'larını + kullanıcının multi-focus doğruluk sorusunu + kalan fizik eksiklerini tek bir sprint'te paketledik. Şirket tarzı karar: Esc-to-cancel için **core'a cancel_check token** threading (mid-scan durdurma), audit logging için `core/audit.py` paylaşımı (v1 ile aynı JSONL konumu, aynı action isimleri), batch dialog için minimal PNG writer (v1'in batch_renderer backend'i halen tam feature; v2 dialog kullanıcı dosyalarını tetikliyor).

## Shipped — v2.0.4
- **Multi-focus accuracy (CEO takip)** — `tests/test_ui2_multifocus.py` (7 test + 1 skip) → (a) sentetik sphere'de find_focus_candidates top-5 içinde true z'ye ~4 step tolerans (empirik, peak shift normal), (b) ranking prominence'a göre, (c) multi-sphere sahne ≥2 peak, (d) ScienceDriver z_min/z_max/metric/n_steps pass-through doğrulandı, (e) effective pixel core'a doğru akıyor.
- **Autofocus z-range + n_steps UI** — `ReconParams.af_z_min_mm`/`af_z_max_mm`/`af_n_steps` + sidebar Autofocus bölümünde 3 widget; `SCHEMA_VERSION` 4→5 bump + v4→v5 migration (-25/+25, 40 backfill). Autofocus/multifocus/QPI batch/depth hepsi `params`'tan okuyor (hardcode bitti).
- **Audit logging** — `core.audit.get_audit_log()` `ScienceDriver` + `ReconstructionDriver`'ın her başarılı job'unda çağrılıyor; action: `reconstruction/autofocus/multi_focus/qpi/qpi_batch/depth_map`; params: `_params_for_audit()` tüm ReconParams alanlarını + sample_id + hologram path + job-specific meta (z_min/z_max/n_steps/metric/prominence/n_sample/n_medium); result_summary: runtime_ms + öznel metrikler. `tests/test_ui2_audit_cancel.py` 3 audit testi (recon/autofocus/QPI).
- **Esc-to-cancel** — `core/autofocus/analysis.py::find_focus_candidates` + `core/depth_map.py::compute_depth_map` fonksiyonlarına `cancel_check: Optional[Callable[[], bool]]` eklendi (core'da boundary'de `AutofocusCancelled` raise). `ScienceDriver._cancel_event` + `cancel()` metodu + dispatch'te cancelled-during-run kontrolü; `ReconstructionDriver` aynı pattern'de. app.py Escape handler → `science.cancel() | driver.cancel()` + status "Cancel requested". 3 cancel testi (cooperative raise, no-inflight noop, late-result discard).

## Shipped — v2.0.5
- **Maximize panels Ctrl+1/2/3/4 + Ctrl+0** — `ZoomableImagePanel.container_tag` eklendi; `_maximize_panel(idx)` diğer üçünü gizler + seçileni `width=-1, height=-1` ile büyütür; `_restore_panel_grid` geri getirir; klavye handler + View menüsüne "Maximize Input/Amplitude/Phase/Info" + "Restore grid" item'ları.
- **Error drawer** — `self._error_log: list[(ts, level, msg)]`, kapasite 64; `_set_status(level="warn"|"danger")` otomatik ekliyor; View → "Errors & warnings…" modal (timestamp'li renkli satırlar + Clear/Close); `_handle_error` callback explicit append. Test: append + cap evict + clear.
- **Line profile tool** — Tools → "Phase line profile (center row)" → Dear PyGui plot dialog'u `phase[H//2, :]` horizontal cut'ı çizer. Full click-drag ROI picker v2.0.6'ya bırakıldı.
- **Batch reconstruct directory** — Tools → "Batch reconstruct directory…" → directory picker → `*.tif/*.png/*.bmp/*.jpeg` bulup her biri için reconstruction çalıştırıyor (effective px + subtract_mean + hann_window tümü current params'tan); output `<stem>_phase.png` + `<stem>_amp.png` olarak kaynak dizine yazılıyor (imageio varsa). Worker thread + mailbox status mesajları; Esc `_driver._cancel_event` ile durdurur.
- **Advanced physics** — `ReconParams`'a 4 field eklendi: `subtract_mean`/`hann_window`/`fft_backend`/`unwrap_method`. Sidebar "Advanced" collapsing header'ında 2 checkbox + 2 combo. `ReconstructionDriver._run` subtract_mean/hann_window'u extract_offaxis'ten önce uyguluyor (v1 batch_renderer paritesi). `ScienceDriver.run_qpi` `UnwrapMethod[params.unwrap_method]` kullanıyor, bilinmeyen isimde `GRADIENT_INTEGRATION`'a fallback. `SCHEMA_VERSION` 5→6 + v5→v6 migration (v1 defaults backfill). 6 advanced test (defaults, migration, roundtrip, hann dims, to_uint8).
- **Run v2.command launcher** — proje kökünde `Run v2.command`, chmod +x; `Hybrid/venv` → `../Phyton/venv` → system Python 3.13 → `python3` ladder; `run_ui2.py`'yi çağırıyor.

## Tests (v2.0.4 + v2.0.5)
25 yeni test:
- `tests/test_ui2_multifocus.py` (7 + 1 skip) — algoritma + plumbing
- `tests/test_ui2_audit_cancel.py` (6) — audit recon/autofocus/QPI + cancel raise + no-inflight + late-result discard
- `tests/test_ui2_advanced.py` (12) — advanced defaults, migration v5→v6, error log cap/clear, to_uint8 edge, hann dims, maximize tags, out-of-range noop

Tüm ui2 test toplam: 68 + 1 skip = **69 testlik ui2 alt-suite** (v2.0.2'deki 37'den geldi).

## Verification
- `PYTHONPATH=src:tests ../Phyton/venv/bin/python -m pytest` → **387 passed, 1 skipped in 10.00s** (v2.0.3'ün 362'sine göre +25, sıfır regresyon)
- `scripts/check_language.py` → exit=0
- Smoke: macOS'ta DhmApp() init ediyor → viewport 1453×945, preview 384, drop_supported=False, effective_px=5.000µm, af_range=[-25, 25], n_steps=40, subtract_mean=True, unwrap=GRADIENT_INTEGRATION, error_log cap=64.

## Kararlar ve gerekçeler
- **Core'a cancel_check geçirmek**: Thread öldürmek Python'da temiz değil; cooperative cancellation boundary'de check + raise, core zaten `AutofocusCancelled` exception'ı taşıyor. Backward-compat için `cancel_check: Optional[...] = None`, eski çağrılar etkilenmez.
- **Audit helper: _params_for_audit()**: tek kaynak, her worker aynı flat dict üretir. Sample ID splice etmeyi v1 `main_window._audit()`'ten aldık — LIMS correlation'ı caller'a bırakmıyor.
- **Batch dialog minimal**: PNG output imageio ile yazılıyor; HDF5/bundle export için v2.0.6 backlog (v1 batch_renderer bu export'u zaten yapıyor, ama v2 dialog şu an PNG preview'a odaklı). Progress status bar'da; dedicated dialog v2.0.6.
- **Maximize `width=-1, height=-1` hack**: Dear PyGui'de gerçek "fullscreen panel" API yok; sibling'leri gizleyip seçileni autosize yaparsak Grid flex'i tek panel'i büyütür. Ctrl+0 orijinal size'ı geri getirir. Clean değil ama pragmatik.

## Açık kalan (v2.0.6+)
- **Line profile click-drag ROI** (v2.0.5'te sadece center row)
- **Batch dialog HDF5/bundle export** + progress table (şu an PNG + status bar)
- **Camera live feed + record** (v1'de tam, v2'de yok — hardware dependency)
- **Crash handler** `sys.excepthook` wire-up
- **High-contrast theme WCAG-AA test** (palette var, compliance test yok)
- **Preset save/load** (user-defined preset dict'i persist)

Kullanıcı "hepsi" dedi ve compliance + UX + physics gap'lerinin hepsi bu sprint'te kapandı. Kalan item'lar yeni feature kategorisinde, v1 paritesi olmayan işler.

---

# Plan — 2026-04-24 — v2.0.6 mega sprint ("Yapın hepsini")

## Context
CEO v2.0.5 review'ındaki "açık kalan" 6 item'a "hepsi" dedi. v1 paritesi kapanmıştı; bu sprint yeni feature + compliance + accessibility polish. Her item için şirket kararı: "doğru olanı yap, kısa yolu değil" — her çıktı test edilecek, her migration backfill eski dump'ları koruyacak, UI/backend ayrık lazy-load disiplinine uyacak.

## Shipped — v2.0.6

- **Crash handler (sys + threading)** — `core.crash_handler` zaten vardı; `install_threading_excepthook()` + `uninstall_crash_handler()` eklendi. Her crash JSON dump'a ek olarak `core.audit.get_audit_log()` kanalına `action="crash"` olarak yazılıyor. `run_ui2.py` entry point'te kurulum; `DhmApp.run()` UI wrapper (toast + error drawer entry + state flush) üstüne gelir ve asıl handler'a chain olur. 7 test: main thread dump, chain, KI skip, thread dump, thread KI skip, audit emit, uninstall.
- **WCAG-AA theme compliance** — `tests/test_ui2_theme_contrast.py` 32 parametrized test (4 palette × 7 rol çifti + white/black sanity + high_contrast AAA). 4 palette fail etti: `dark.danger` (3.70), `light.accent` (4.44), `light.success` (4.33), `light.warn` (3.06). Minimal renk değişimleri ile fix: `(240,110,125)`, `(36,100,180)`, `(25,110,40)`, `(140,100,0)`. Tüm palettes AA (text ≥4.5); high_contrast AAA (≥7).
- **User-defined preset save/load** — `Ui2State.user_presets: dict[str, dict]`, `SCHEMA_VERSION` 6→7, `_v6_to_v7` migrator. `DhmApp._presets()` instance method artık built-in + user presets merge eder; user preset built-in adını shadow edemez. Sidebar Preset bölümüne "Save preset…" + "Delete…" button'ları; `PresetChips.rebuild(labels)` dinamik liste desteği. State'e persiste edilir. 13 test: schema default, v6 migration, corruption defense (list → {}), nested non-dict drop, roundtrip, merge order, collision reject, delete rejects built-in, delete removes entry, missing no-op, clear selected, snapshot physics fields, built-in/tuple sync.
- **Line profile click-drag ROI** — `DhmApp._sample_line(phase, p1, p2, n=512)` scipy `map_coordinates(order=1, mode="nearest")` bilinear sampling; horizontal/vertical/diagonal/out-of-bounds hepsi test edildi. Tools menüsüne "Phase line profile (draw…)" item; click/release mouse handler'ları phase panel hover kontrolü ile gated. Esc line mode'u cancel'lar (cancel-job path'inden önce). 12 test: sampling invariants (5), mode gating (3), full gesture simulation (1), degenerate case warn (1), param (3).
- **Batch HDF5 bundle + progress table** — `src/core/batch_bundle.py` yeni modül: `BatchEntry` dataclass + `write_batch_hdf5` + `read_batch_hdf5` (schema v1, gzip-4 compression, deterministic stem-collision suffix). `app._open_batch_dialog` per-file progress table (mvTable) ile modal açıyor; mode combo "PNG per file / HDF5 bundle / Both". Worker mailbox'a `("batch_row", (idx, status, rt, level))` post ediyor; mevcut mailbox dispatcher row color'u güncelliyor. Bundle schema: root attrs (schema_version, app_version, sample_id, created_utc, n_holograms, recon_params_json), `/holograms/<safe_stem>/{phase, amplitude}` + per-group attrs. 7 test: empty raises, single roundtrip, multi order, collision _02 suffix, non-serialisable metadata stringified, safe key slugification, parent dir creation.
- **Camera live feed + TIFF stack recording** — `src/ui2/camera_feed.py` yeni modül: `CameraSource` Protocol + `SyntheticCamera` (512px fringe pattern, 30 fps target, deterministic RNG) + `TiffStackRecorder` (lazy tifffile, uint16 append, bigtiff, append-on-write) + `AcquisitionThread` (daemon, chunked sleep for <100ms stop latency, on_frame/on_fps callbacks, on_frame exception isolation). Tools menüsüne "Camera live feed (start/stop)" + "Record camera to TIFF stack…". Live frame mailbox'a `("camera_frame", np.ndarray)` + `("camera_fps", float)`. Main loop `panel_input.set_image(frame)` ile live paint; kayıt aktifse status bar "Camera: X.Y fps · recording N frames". `run()` finally'sı `_on_camera_stop` çağırarak TIFF'i temiz kapatıyor. 12 test: Protocol conformance, frame shape/range, animation diff, grab-before-start tolerance, recorder roundtrip + error cases, thread callback fires ≥3/2s, stop <0.5s, recorder forward, exception isolation.

## Tests (v2.0.6)
83 yeni test; toplam **470 passed, 1 skipped in 10.47s**.
- `tests/test_crash_handler.py` (7)
- `tests/test_ui2_theme_contrast.py` (32 → 4 palette × 7 pair + 4 sanity)
- `tests/test_ui2_user_presets.py` (13)
- `tests/test_ui2_line_profile.py` (12)
- `tests/test_batch_bundle.py` (7)
- `tests/test_ui2_camera_feed.py` (12)

v2.0.5'ten gelen 387 + bu sprint 83 = **470 passed**, sıfır regresyon.

## Verification
- `PYTHONPATH=src:tests ../Phyton/venv/bin/python -m pytest` → **470 passed, 1 skipped in 10.47s**
- `scripts/check_language.py` → exit=0
- Smoke: macOS'ta DhmApp init → `user_presets=[]`, `camera_thread=None`, `line_mode=False`, advanced defaults intact.

## Kararlar ve gerekçeler
- **`ui2/__init__.py` lazy + `theme.py` lazy import** — test order 470'in altından geçebiliyor çünkü herhangi bir ui2 alt modülü import edildiğinde Dear PyGui C extension'ı yüklenmiyor; cross-file stub conflict'i bitmiş durumda.
- **WCAG fix minimal delta** — hue korundu, sadece L* ayarlandı, app görsel kimliği değişmedi. AA dışında high_contrast AAA gereği ayrıca test ediliyor (accessibility kullanıcıları için strict commitment).
- **User presets: built-in shadow reject** — dict merge'te built-in name'e düşen user preset silinir (save path zaten UI'da uyarı veriyor). `Ui2State` corruption defense: list / scalar gelirse `{}`'a reset — hand-edit riski.
- **Line profile: map_coordinates(mode="nearest")** — out-of-bounds endpoint'ler edge'e clamp, NaN yok, kullanıcı ekran dışına çizebilir.
- **Batch HDF5 stem collision**: `_safe_group_key(stem, taken)` + slug + `_NN` suffix; hiç silent overwrite yok.
- **Camera: synthetic default, hardware opt-in**: v1 NICamera Windows'a bağımlı; v2 default'u SyntheticCamera, real hardware backend `CameraSource` Protocol implement ederek sonradan bağlanabilir. `AcquisitionThread.run()` on_frame exception'ı log+devam eder — buggy UI handler feed'i öldürmez.
- **TIFF stack uint16**: float32 4× büyük; uint16 dinamik aralığı koruyor (65535 levels) ve lab için yeterli. BigTIFF flag uzun session'lar için (>4GB).

## Açık kalan (v2.0.7+)
- Real hardware camera backends (Pylon, IDS, Thorlabs) için `CameraSource` implementations
- MP4 recording (tifffile yerine imageio-ffmpeg)
- Line profile: multi-line (aynı frame'de N tane), profile karşılaştırma dialog
- Batch dialog: "Resume from last" (sig hash ile aynı param'larla aynı output varsa skip)
- User preset dialog: context-menu "Edit existing" (şu an save ile aynı isim overwrite)
- Audit log viewer dialog (şu an sadece JSONL diskte)

Kullanıcı tekrar "hepsi" derse bunlar gelir; aksi halde v1 paritesi + compliance + accessibility + workflow tool'ları hepsi kapalı. Pilot için ship edilebilir durumda.
(oturum sonunda doldurulacak)

---

# 2026-04-24 akşam — 5-bug acil toplantı (pilot feedback)

## Context
Kullanıcı pilot'ta aşağıdakileri rapor etti: (1) hologram ters çevriliyor, (2) referans çıkarma çalışmıyor, (3) ilk yüklemede input paneli boş, (4) autofocus yavaşladı, (5) arayüzde hala scroll gerekli.

## Shipped (synthetic olarak doğrulandı, 505 pytest yeşil)
- **Bug #3 — ilk-load input panel boş** — kök sebep: `_load_hologram` `_push_texture("tex_input", …)` literal string geçiyordu, `_panel_for_tex` random UUID ile karşılaştırıyordu → match fail → sessiz fallback → texture hiç push edilmiyordu. 3 call site düzeltildi (input + phase + depth overlay), `panel_input.tex_tag` (gerçek UUID) kullanılıyor artık.
- **Bug #2 — reference subtract çalışmıyor** — kök sebep: `ReconstructionDriver._run`'da vardı ama `workers._prepare_field`'a (autofocus/multi-focus/QPI/depth ortak path'i) hiç kopyalanmamıştı. İki paylaşılan helper çıkardım: `_preprocess_raw()` (subtract_mean + hann + normalize) ve `_extract_field_with_reference()` (offaxis + reference-division). Her iki pipeline path'i artık aynı preprocessing + referans mantığını kullanıyor.
- **Bug #5 — scroll hala görünüyor** — kök sebep: `_tier_for_width()` sadece genişlik ekseninden tier seçiyordu; 1440×800 laptop'ta 384 tier seçiliyordu ama dikeyde 2×(384+40)+144=992 > 800 → scrollbar. `_tier_for_size(w, h)` iki eksene birden bakıyor artık, viewport yüksekliği 1000→1150 cap, ekran kullanımı 85%→90% yükseltildi.
- **Bug #1 — hologram ters** — bazı sensör/kamera TIFF'leri bottom-up row order ile yazıyor. `View → Flip display vertically/horizontally` checkbox item'ları eklendi — display-only, pipeline math'e dokunmuyor. User toggle ile kendi yönünü seçer.

## Synthetic regression suite (tests/test_ui2_v207.py, 14 test)
- Reference subtract armed iken extracted field değişiyor (referansız vs referanslı karşılaştırma)
- `subtract_reference=False` iken extractor dokunmuyor
- `_preprocess_raw` subtract_mean + hann_window branch'leri beklenen gibi davranıyor
- `_load_hologram` hala `panel_input.tex_tag`'i kullanıyor (literal string yok)
- `_tier_for_size` 5 farklı (w,h) için doğru tier döndürüyor + yatay/dikey invariant'lar
- `_tier_for_width` legacy alias hala çalışıyor
- Flip toggle state geçişleri + hologram yoksa repaint crash yok
- Workers module'de `_flip_display_*` referansı yok (bilimsel path'e sızmıyor)

## ⚠️ Hatırlatma — kullanıcı kontrolü gereken
- **Bug #4 (autofocus yavaşlığı)**: Core benchmark 512×512 × 40 step zscan = 265ms (n=1.0) vs 279ms (n=1.33). Core yavaş değil. User'ın gözlemlediği yavaşlama dispatcher overhead / adaptive algoritma tercihi / gerçek hologram boyutuyla ilgili olabilir. **Gerçek senaryo bilgisi istiyoruz**: hangi algorithm seçili? n_steps kaç? Hologram shape? TIFF load süresi mi, scan süresi mi uzun? Bu rakamları aldığımızda kesin fix yazılır. Şimdilik pin'lenmedi.
- **Bug #1 flip son doğrulama**: sentetik test state toggle'ı kontrol ediyor. Gerçek TIFF'inde önce "Flip display vertically" sonra horizontal'ı dene, doğru orientation hangisi sabitleyelim. Eğer sensör/kamera hep aynı orientation'da yazıyorsa preset default'una alırız.
- **Bug #5 scroll son doğrulama**: senin ekran boyutunda tier kaç seçiliyor kontrol et (View'da preview size görünmüyor ama status text'e debugger'ımız koyabilir istersen). 1440×800 MacBook Air → tier 288 bekleniyor, scroll olmamalı.

## Sıra (kalan sprint için)
Kullanıcı müsait olunca:
1. Autofocus yavaşlık gerçek senaryo ölçümü (bug #4)
2. Flip doğru yönün hangisi olduğunun doğrulaması (bug #1)
3. Scroll'un mevcut ekranda hala görünüp görünmediği (bug #5 manuel doğrulama)

## Review
- 505 pytest pass, sıfır regresyon
- Bug #2, #3, #5: synthetic olarak pin'li, deterministik
- Bug #1: display toggle + sentetik state kontrolü
- Bug #4: user feedback bekliyor

---

## 2026-04-24 — End-to-end autofocus→measure testleri (lateral + depth)

User sorusu: "lateral uzunluk ve depth correction testi yaptın mı, sanal hologramlarla (auto focus sonrası)"

Önceki sanal testler hep truth z'ye propagate ediyordu — otofokus çıkışını kullanmıyordu, yani end-to-end değildi. Eksikti.

### Yapılan
`tests/test_focus_validation.py` iki yeni test grubu:
- `test_end_to_end_autofocus_then_measure_size` (3 parametre: z = 12, 16, 20 mm; r = 15 µm; Δn = 0.07)
- `test_end_to_end_two_objects_via_multifocus` (2 küre, 12 ve 17 mm)

Her iki test de:
1. Hologramı sentezle
2. Off-axis extract
3. `autofocus_zscan` / `find_focus_candidates` çalıştır — **truth z'yi kullanma**
4. Reconstruct at z_est (not truth)
5. Assert:
   - z_est within 5–7× scan step of truth (depth correction)
   - reconstruction finite + non-trivial phase range (NaN regression guard)
   - OPD peak-to-peak ∈ sphere-centred tight disk ≈ min(2·r·Δn, λ) ±70 %

### Lateral diameter assertion neden atıldı
Single-sphere sentetik hologramda Δn = 0.07 için fazın 3–4 kez sarılması + off-axis carrier residual her yerde aynı amplitude'da beat deseni yaratıyor. Her area-above-threshold metriği sphere contour'u yerine wrap lobe'larını + ringing'i ölçüyor → 3–5× truth. Pixel-local istatistikler (std, var) window konumundan bağımsız nearly eşit. Triple-sphere test'inde metrik çalışıyor çünkü multi-object sahne background'a structure ekliyor — single sphere'de o structure yok.

Dürüst alternatif: OPD'yi *sphere'in expected pixel*'inde dar bir disk içinde ölç. Lateral pozisyon kontrolü (yanlış pixel'a land ederse disk all-background olur → swing çok düşük) + depth correction (OPD ≈ λ ± 70 %) tek assertion'da. Lesson log'a kaydedildi.

### Durum
- 17 focus_validation test pass (3 yeni single-sphere + 1 yeni multi-focus dahil)
- Full suite: 512 pass, 1 skipped, sıfır regresyon
- Depth correction sentetik olarak pinli ✅
- Lateral presence (sphere expected pixel'ında OPD varlığı) pinli ✅
- Precision lateral diameter: single-sphere sentetik'te reliable değil; gerçek donanım veya multi-object sahne ile user doğrular

---

## 2026-04-24 — Bug #4 (autofocus yavaşlığı) — proaktif benchmark + pin

User gerçek senaryo vermeden çözüm yazmak için körsün. Önce ölçüm, sonra karar.

### Benchmark sonuçları (`scripts/bench_autofocus.py`, ENTROPY, 40 step)

| shape | zscan | coarse_to_fine | robust | adaptive_gradient | adaptive_bracketing | adaptive_distance |
|-------|------:|---------------:|-------:|------------------:|--------------------:|------------------:|
| 256²  |   65  |    95          |   120  |        66         |      88             |     23            |
| 512²  |  250  |   360          |   460  |       255         |     340             |    240            |
| 1024² | 1070  |  1530          |  1920  |      1080         |    1430             |    990            |
| 2048² | 4280  |  6170          |  7710  |      4300         |    5730             |   4060            |

Tüm rakamlar ms. Dispatcher overhead (TIFF load + preprocess + offaxis extract) 512² için **14 ms** — core time'a karşı negligible, regresyon yok.

### User'ın "5 saniye" anısı
2048×2048 × 40 step zscan = **4.3 sec**. Tam eşleşme. Büyük ihtimalle user'ın gerçek senaryosu 2048² hologramlar ile zscan. "Yavaşladı" algısı için olası sebepler:
1. Farklı algoritma seçmiş (robust = 7.7 sec, coarse_to_fine = 6.2 sec @ 2048²) — "önceden zscan'di, sonradan robust'a geçti" senaryosu en mantıklı
2. `n_steps` artmış (40 → 80 = 2× slowdown)
3. Hologram boyutu büyümüş (512 → 1024 = 4×, 1024 → 2048 = 4×)

Hiçbiri kod regresyonu değil — user-side configuration drift. Gerçek senaryo gelince kesinleşir.

### Regresyon koruma
`tests/test_autofocus_speed_baseline.py` eklendi — zscan 256/512/1024 için ceiling'ler (200 / 800 / 3000 ms) 2× mevcut bench'e set edildi. `_make_fast_evaluator`'daki FFT cache kırılırsa (her z için yeniden fft2), runtime ~2× olur → bu test patlar.

### Benchmark script pinli
`scripts/bench_autofocus.py` — `--shapes 256,512,1024,2048` argümanı. User dönüp "senaryom 2048² × 80 step robust" dediğinde hemen `--shapes 2048` + `n_steps` düzenleyerek rakamı çıkarırız.

### Durum
- Core fast_evaluator path: FFT cache aktif (`_make_fast_evaluator` field_spectrum'u bir kez hesaplıyor, her evaluate(z) onu reuse ediyor) ✅
- Dispatcher overhead: 14 ms @ 512² (negligible) ✅
- Baseline pin: 3 parametrize test green ✅
- 6 algoritma × 4 boyut bench JSON'a pinli (`tasks/bench_autofocus.json`) ✅
- ~~`find_focus_candidates` (multi-focus button) `propagate(force_python=True)` kullanıyor~~ — 2026-04-24 içinde çözüldü; aşağıdaki multi-focus refactor kaydına bak.
- User'dan beklenen: gerçek senaryo (hologram shape, algorithm, n_steps, TIFF boyutu) → kesin fix path

---

## 2026-04-24 — Multi-focus refactor: `_make_fast_evaluator` ortaklığı

### Motivasyon
Bug #4 çalışmasında fark ettim: `find_focus_candidates` ve `scan_metric_landscape` `propagate(force_python=True)` döngüsü kullanıyordu; 6 autofocus algoritmasının hepsi `_make_fast_evaluator` kullanıyordu. Tek code path olsa, o evaluator'a eklenecek her perf win multi-focus'a da otomatik akar.

### Yapılan
`src/core/autofocus/analysis.py::find_focus_candidates` — inner döngü `_eval = _make_fast_evaluator(...)` pattern'ine geçirildi. Semantic aynı (aynı peak'ler, aynı prominence); sadece propagation path değişti.

### Perf ölçümü (60 steps, ENTROPY, Δn=0.07 sphere)

| shape | before (ms) | after (ms) | delta |
|-------|------------:|-----------:|------:|
| 256²  |       101   |     123    |   +22 |
| 512²  |       406   |     392    |   -14 |
| 1024² |      1594   |    1598    |    +4 |

Perf-wise **wash** — `propagate()`'in `spectrum.copy()`'si 1024² complex64 için sadece ~0.16 ms per call, 60 iter = ~10 ms total, %0.6 overhead. FFT + metric compute dominant. Ölçümden önce "25-30% win" tahmin etmiştim, yanılmıştım. Yorum düzeltildi.

### Gerçek kazanç: single path
6 autofocus algoritması + multi-focus artık aynı `_make_fast_evaluator`'a bağlı. Gelecekte o evaluator'a eklenen her optimizasyon (batch FFT, GPU backend, ROI fast-path) multi-focus'a otomatik ulaşır. Ayrıca `_GLOBAL_RECON_CACHE` dependency kalktı — thread safety / test isolation daha sağlam.

### Regresyon koruma
`tests/test_autofocus_speed_baseline.py`'a 3 multi-focus parametrize test eklendi (256/512/1024 ceiling 300/1200/3500 ms). `_make_fast_evaluator` cache'i kırılırsa hem zscan hem multi-focus testleri birden patlar — regresyonu tek fonksiyona pin'ler.

### Durum
- 518 pytest pass (+3 multi-focus speed baseline, önceki 515'ten), 1 skipped, sıfır regresyon
- Autofocus + multi-focus aynı code path'i kullanıyor artık ✅
- Correctness: 56 `find_focus_candidates` test'i green (focus_validation + ui2_multifocus + focus_candidates_dialog + stress_holograms)

---

# 2026-04-27 — Bug regression aracı + Combined roadmap + v2.0.7 sprint açılışı

## Context
User isteği: "şimdiye kadar hata aldığımız her şeyi test etmek için tool oluştur, sonra devam et". Eski kararlar (v2.0.7+ açık backlog, v2.0.3 + v2.0.6 review notları, multi-focus refactor açtığı kapı, 4 sprint candidate) + Lindqvist Lab toplantısının çıktısı tek roadmap'te birleştirildi.

## Ship 1 — `scripts/check_bugs.py` (Bug regression tool)
- 35 bug entry'lik registry: bug_id, fix-date, topic, status (test/lesson_only/manual), test node id
- Subprocess pytest runner, bug-by-bug pass/fail/skip/lesson/manual table
- ANSI renkli output + `--json` (CI için) + `--filter B-029` (tek bug) + `--only test/lesson_only/manual`
- Exit 0 = her testable bug pass, exit 1 = en az bir regresyon
- İlk run: **22 PASS / 0 FAIL / 0 SKIP / 8 LESSON / 5 MANUAL**

## Ship 2 — `tasks/roadmap.md` (Combined roadmap)
v2.0.7 → v3.0 birleşik plan. Maddeler:
- **v2.0.7** Time-lapse foundation (3 hafta) — Session model, headless CLI, multi-user, preset save/load, audit viewer, batch resume
- **v2.0.8** Tracking + calibration (3 hafta) — drift correction, per-cell tracking, NIST bead workflow, multi-line profile
- **v2.0.9** Paper-ready (2 hafta) — vector PDF, Zenodo bundle, line profile ROI, crash handler, WCAG-AA
- **v2.1.0** GPU + headless (4 hafta) — PyTorch backend, batch FFT, ROI fast-path, Linux Docker
- **v2.1.x** Real hw (3 hafta) — Pylon/IDS/Thorlabs, MP4 recording
- **v3.0** AI seg (6-8 hafta) — Cellpose, cell-cycle classifier, onboarding wizard

Eski backlog item'ları (drop zone live indicator, export highlight, line profile ROI, batch HDF5, crash handler, WCAG-AA, audit viewer, preset save/load, real hw cameras, MP4, multi-line, batch resume, preset edit) hepsi v2.0.7-v2.1.x içine yerleştirildi.

## Ship 3 — v2.0.7 T4: Preset Edit/Replace flow + archive
v2.0.7+ backlog'unun "User preset dialog 'Edit existing'" maddesi. v2.0.6'da save+delete vardı, edit flow yoktu — aynı isim sessiz overwrite ediyordu.

### Yapılan
- `Ui2State.user_preset_archive: dict[str, list[dict]]` field, `SCHEMA_VERSION` 10→11
- `_v10_to_v11` migration (state_store.py) + Qt-side `_migrate_v10_to_v11` (settings_store.py)
- `_hydrate_ui2` archive corruption defender (list yerine scalar, list içinde non-dict)
- `DhmApp._save_user_preset(name, dict, *, archive_previous=None)` tek source of truth
- `DhmApp._open_replace_preset_dialog` — collision'da side-by-side diff modal
- `DhmApp._commit_replace_preset` — Replace branch: eski snapshot archive'a, yeni dict user_presets'e
- 10-version cap per name (oldest dropped on append)

### Test (`tests/test_ui2_user_presets.py`)
+10 yeni test (toplam 23):
- v10 → v11 migration backfill
- Corrupted archive normalises to {}
- Non-list / non-dict entries dropped
- Roundtrip preserves history
- `_save_user_preset` no-archive new entry
- `_save_user_preset` with archive_previous appends
- 10-version cap enforced
- Cross-name archive isolation

## Ship 4 — v2.0.7 T0: Session data model
Lindqvist Lab toplantısının #1 maddesi (Karin'in 3000-hologram pain'i). Time-lapse pipeline'ın omurgası.

### Yapılan
`src/core/session.py`:
- `HologramFrame` (frozen): `path, timestamp_s, index, params_overrides, notes`
- `Session`: `id (UUID), created_at (ISO-8601 UTC), operator, sample_id, root_dir, params, frames`
- Factories: `Session.new(...)`, `Session.from_directory(dir, glob_pattern, sort_by)`
- Mutation: `add_frame`, `with_params`, `effective_params_for` (frame override > session default)
- Path resolution: `resolve_frame_path` — relative anchors against root_dir, absolute pass through
- `signature()` deterministic hex digest (params + frame paths + sizes) for `--resume-if-exists`
- Atomic JSON: same-dir tempfile + `os.replace`, malformed JSON raises ValueError
- `to_dict` / `from_dict` tolerant (missing fields default, invalid frames dropped)

### Test (`tests/test_session.py`)
30 test:
- Construction (UUID uniqueness, from_directory glob/sort)
- Mutation (add_frame indexing, params merge, override semantics)
- Path resolution (relative + root_dir, absolute pass-through)
- Signature determinism + invalidation triggers (params change, frame add, file size change)
- JSON roundtrip (atomic, malformed, missing-field tolerance, invalid-frame skip)

## Durum
- **558 pytest pass** (518 → +40), 1 skipped, sıfır regresyon
- Bug registry: 22 PASS / 0 FAIL / 0 SKIP / 8 LESSON / 5 MANUAL ✅
- v2.0.7'nin 6 sprint maddesinden 2'si tamamlandı (T0 Session model + T4 preset edit)
- Sıradaki: T1 headless CLI (5 gün) — `python -m dhm.session run session.json --out results/`
- Ardından: T2 per-frame CSV, T3 multi-user profile, T5 audit viewer, T6 batch resume

---

# 2026-04-27 (geç akşam) — Per-phase bug-test ritueli + v2.0.7 T1/T2/T3/T5/T6 ship

## Context
İki hat paralel ilerletildi:
1. User isteği: "her phase için bug test toolları üret, lessons.md'ye yaz". Tek `check_bugs.py` yetersizdi → registry data-only modüle çıkarıldı, runner ortaklandı, **10 per-phase wrapper** eklendi (4 aktif + 6 gelecek).
2. Sprint maratonu: v2.0.7'nin kalan 5 maddesi (T1 CLI, T2 CSV, T3 multi-user, T5 audit viewer, T6 resume) tek oturumda kapatıldı.

## Ship 5 — Per-phase bug regression altyapısı
- `scripts/bug_registry.py` — `Phase` enum (10 değer) + `BugEntry` (artık `phase` field'lı) + `BUG_REGISTRY` (35 entry, hepsi backfill'lendi)
- `scripts/_bug_runner.py` — paylaşılan runner, `run_phase()` + `run_all()`, ANSI tablo + JSON çıktı, phase column toggle
- `scripts/check_bugs.py` — `--phase <name>` filter desteği eklendi
- `scripts/check_bugs_phase_*.py` × 10 — 5-satırlık wrapper'lar:
  - Aktif: pre_pilot, ux_patch, dpg_port, pilot_patches
  - Boş iskelet: timelapse, tracking, paper_ready, perf_gpu, hardware, ai
- `tasks/lessons.md § 2026-04-27` — kural pin'lendi: "Her phase kendi bug regression tool'unu yaratır"
- `tasks/roadmap.md § Sprint cycle ritüeli` — adımlar güncellendi, "yeni phase başlatmak" rehberi eklendi

## Ship 6 — v2.0.7 T3: Multi-user profile
- `src/core/user_profile.py` — `current_user()` (DHM_USER env > getpass > "default"), `sanitise_username()` (kebab-clean, dash-only fallback), `user_state_dir/path()` per-user, `migrate_legacy_state_if_needed()` legacy → users/<x>/, `list_known_users()`, `set_root_dir()` test seam.
- `src/core/audit.py` — record artık `operator` field'ını da yazıyor (back-compat: `user` aynen kalır).
- `src/ui2/state_store.py` — `default_state_path()` her load/save'de kullanılıyor; legacy migration ilk run'da otomatik.
- `tests/test_user_profile.py` — **23 test**: sanitise edge cases, env override, getpass fallback, per-user isolation, list_known_users sorting, legacy migration (3 senaryo), audit operator embedding (2 senaryo).

## Ship 7 — v2.0.7 T2: Per-frame CSV export
- `src/core/session_export.py` — `CellMeasurement` + `FrameResult` dataclass'ları, `write_session_csv(layout="long"|"wide")`, custom `wide_metrics`.
  - Long format: row per (frame, cell); cell-less frames hala bir satır alır (autofocus-only branch).
  - Wide format: row per frame, columns `cell_<id>_<metric>`; missing cells blank.
  - `_fmt(None)` → `""` (Excel'de blank, "None" değil).
- `tests/test_session_export.py` — **11 test**: long/wide ayrımı, session metadata, error field, blank-cell missing metric, custom metrics, layout error, nested-path mkdir, empty-result header-only.

## Ship 8 — v2.0.7 T1 + T6: Headless CLI runner with resume
- `src/cli/run_session.py` — `run` + `inspect` subcommands.
  - `run` flow: `Session.load_json` → frames iter → load_any → `_preprocess_raw` → off-axis extract → `autofocus_zscan` → optional propagate → `frame_<i>.json` + signature marker → `session.csv` aggregate.
  - `--phase autofocus|reconstruct|all`, `--workers N` (ProcessPoolExecutor), `--csv-layout long|wide`, `--quiet`.
  - SIGINT handling: graceful, in-flight frame finishes, signature persisted, exit 130.
  - **T6 resume**: per-frame JSON carries `session_signature`; `--resume-if-exists` skips when sig matches. Mismatch → re-run (param drift catches stale outputs).
  - JSONL progress to stdout: `session_start`, `frame_done` × N, `session_done`. Tail -f friendly.
  - Audit log: `session_start` + `session_done` events.
- `src/cli/__init__.py` — package marker.
- `tests/test_cli_session.py` — **13 test**: inspect, end-to-end run (per-frame JSON + CSV + signature), JSONL progress events, --quiet stdout suppression, resume happy path, resume re-run on signature mismatch, resume mid-crash partial output, phase filter (autofocus / reconstruct), error containment (missing source TIFF), argparse `main()` shape (3 senaryo).

## Ship 9 — v2.0.7 T5: Audit log viewer
- `src/core/audit_viewer.py` — pure data layer. `AuditEntry` (parsed dict + raw stash), `iter_entries()` newest-first across daily files + within file, `read_entries(limit=)`, `AuditFilter` (operator + action + free-text query), `apply_filter`, `known_operators/actions` for dropdowns. Stream-friendly (one line at a time), tolerant to malformed JSONL (skip + continue).
- `src/ui2/dialogs.py::show_audit_viewer` — DPG modal: operator + action combos (populated from log itself), free-text query box (Enter triggers refresh), 4-column resizable table (timestamp / operator / action / params preview), 500-entry cap.
- `src/ui2/app.py` — Help → "Show audit log…" menu item + `_show_audit_viewer` callback.
- `tests/test_audit_viewer.py` — **18 test**: list_log_files (sort, ignore non-jsonl, missing dir), iter_entries (within-file order, across-file order, malformed-line tolerance, date range), AuditEntry from_dict tolerance, AuditFilter (operator, action, free-text query, combined intersect, empty), known_operators/actions deduplication.

## Rakamlar
- **623 pytest pass** (558 → +65), 1 skipped, sıfır regresyon
- Bug registry: hâlâ 22/22 PASS ✅
- v2.0.7'nin **6 sprint maddesinden 6'sı tamamlandı** — T0+T1+T2+T3+T4+T5+T6 hepsi ship
- Per-phase wrapper'lar 4 aktif + 6 future, hazır iskelet

## v2.0.7 sprint kapanış kararı
Lab demo için artık hazır:
- Karin: `python -m cli.run_session run session.json --out /tmp/out --workers 4` → 3000-frame overnight session, ertesi sabah CSV
- Sven: Help → Show audit log… → operator filter → "yesterday's runs by Erik" 1 saniyede
- Erik: kendi `~/.dhm-reconstruction/users/erik/ui2_state.json`'ı Karin'inkini etkilemiyor
- DHM_USER env override: shared workstation senaryosunda CI/headless run için gerçekçi

## Sıradaki — v2.0.8 başlangıcı
Roadmap'e göre: Tracking & calibration sprint'i.
- Drift correction (phase correlation registration)
- Per-cell tracking (trackpy.link_df entegrasyonu)
- NIST 10 µm bead calibration workflow
- Multi-line profile (v2.0.7+ backlog'undan)

`scripts/check_bugs_phase_tracking.py` boş iskelet hazır — ilk bug yakalandığında registry'ye `Phase.TRACKING` tag'iyle ekleriz.

---

# 2026-04-27 (gece) — v2.0.8 sprint ship: Tracking & calibration

## Context
"Devam et" → v2.0.8'in tam paketi tek oturumda kapatıldı.

## Ship 10 — D1 Drift correction (`src/core/registration.py`)
Phase correlation tabanlı kayma estimatörü. F_a · conj(F_b) → normalised cross-power → IFFT → integer peak + parabolic 3-pt sub-pixel fit. ``DriftEstimate(dy_px, dx_px, peak_corr)`` döner; ``apply_drift`` integer translate (zero-fill); ``drift_track_session`` cumulative drift across N frames. Trackpy yerine pure scipy.fft kullanıldı — external dep yok.
- **19 test** (`tests/test_registration.py`): integer shift recovery 6 senaryo, sub-pixel via Fourier synth, noise robustness, apply_drift round-trip, session cumulative, edge cases (mismatched shapes, max_shift_px window, low-correlation sentinel).

## Ship 11 — D2 Per-cell tracking (`src/core/tracking.py`)
Trackpy yerine scipy.optimize.linear_sum_assignment ile Hungarian-algorithm tabanlı linker. ``Detection(cy_px, cx_px, frame_idx, payload)`` → ``link_detections(per_frame, max_distance_px)`` → ``Track(cell_id, detections)``. Birth/death events: yeni cell → yeni ID, kayıp cell → track sonu, gap'ten dönen cell → yeni ID (re-ID v3.0 territory). ``detections_from_clusters`` adapter `core.depth_map.ClusterHeight` üstünden tracking'e bağlanır.
- **16 test** (`tests/test_tracking.py`): single cell stability, two cells separate, Hungarian crossing-paths optimality, birth/death bookkeeping, distance threshold, edge cases, payload carry-through, cluster adapter (canonical + alt attr names).

## Ship 12 — D3 NIST calibration workflow (`src/core/calibration.py`)
Karin'in NIST 10µm bead haftalık check'i artık tool-driven. ``measure_bead_diameter`` `sin(phase)` + area-threshold equivalent disk; ``classify(drift)`` traffic-light (green < 2 %, yellow 2-5 %, red > 5 %); ``record_check`` per-user `calibration_history.jsonl` append + audit emit; ``load_history`` tolerant reader (malformed lines skip, partial records default-fill).
- **24 test** (`tests/test_calibration.py`): 8 classify parametrize, custom thresholds, diameter recovery from synthetic phase, off-centre bead, blank input, history append + load + JSONL roundtrip, explicit operator/path overrides, malformed-line tolerance.

## Ship 13 — D4 Multi-line profile (`src/core/line_profile.py`)
v2.0.5'in single-line profile'ının çoklu-line evrimi. ``LineProfile(y0,x0,y1,x1,label,colour_rgb,n_samples)``, bilinear interpolasyonla sub-pixel sampling (out-of-bounds → NaN/sentinel), auto n_samples (~1 sample/pixel), ``stats_for`` NaN-safe min/max/mean/std. DPG dialog wrapper future work; pure-data sampler şimdi.
- **15 test** (`tests/test_line_profile.py`): horizontal ramp, distance axis, diagonal sub-pixel bilinear, auto-N, zero-length, OOB NaN, custom fill_value, error path, sample_profiles order, NaN-safe stats, geometry helpers.

## Rakamlar
- **697 pytest pass** (623 → +74), 1 skipped, sıfır regresyon
- v2.0.8 sprint **4 maddeden 4'ü ship** (D1 + D2 + D3 + D4)
- Bug registry: 22/22 PASS hâlâ ✅
- `scripts/check_bugs_phase_tracking.py` hâlâ "no bugs registered yet" — sprint clean ship

## v2.0.8 → v2.0.9 geçiş
Roadmap sıradaki: **v2.0.9 — Paper-ready output** (2 hafta).
- Vector PDF export (matplotlib backend, scale bar + colorbar + ticks)
- Zenodo-ready bundle (figures.pdf + raw_data.csv + params.json + checksum + README)
- Line profile click-drag ROI dialog (`core.line_profile` zaten hazır, UI kalan iş)
- Crash handler `sys.excepthook` wire-up
- WCAG-AA contrast compliance test
- Report mode export buttons highlight (v2.0.3 backlog'tan)
- Drop zone reconstruction live indicator (v2.0.3 backlog'tan)

---

# 2026-04-27 (geç gece) — v2.0.9 sprint ship: Paper-ready output

## Context
v2.0.8 kapanır kapanmaz "Devam et" → v2.0.9 ana eksenleri (P1-P4) tek oturumda.

## Ship 14 — P1 Vector PDF report (`src/core/pdf_report.py`)
matplotlib + `backend_pdf.PdfPages` → tek-sayfa vector PDF. `PdfReportData` dataclass'ı opsiyonel her field (header, hologram, phase, amplitude, line_profiles, calibration footer). Layout: header / üst görüntüler / line profile overlay + recon params block / autofocus + QPI + calibration footer. Scale bar (10 µm beyaz) ve per-image colorbar otomatik. Image rasterized@300dpi (PDF içinde küçük) ama eksen + scale bar vector. matplotlib'i lazy-import ediyor → import-time cost minimal.

## Ship 15 — P2 Zenodo bundle (`src/core/zenodo_bundle.py`)
Citation-ready supplementary zip: `figures.pdf` + `raw_data.csv` + `params.json` + `checksum.txt` + `README.md`. SHA-256 her dosya için, 64KB chunk'larla streaming hash (büyük PDF için RAM-friendly). README markdown: sample id + operator + timestamp + reproduce talimatı. ``BundleSpec.extras`` opsiyonel attachment dict (ör. reproduce.py script).

## Ship 16 — P3 Crash handler v2 wiring
Mevcut `core.crash_handler` v1 entry point'inde (`src/main.py`) installuydu, v2 entry point'inde değildi. `src/ui2/app.py::main()` artık `install_crash_handler()` + `install_threading_excepthook()` çağırıyor — DhmApp'in UI-side wrapper'ı (`_install_ui_crash_wrapper`) bu base hook'a chain'liyor → crash dump JSON + UI toast + state flush üçü birden fire ediyor.

## Ship 17 — P4 WCAG-AA contrast audit (`src/ui2/wcag.py`)
Programmatic WCAG 2.1 contrast checker:
- `relative_luminance(rgb)` — sRGB gamma-decode + Y matrix per WCAG Appendix A
- `contrast_ratio(fg, bg)` — symmetric, [1, 21]
- `audit_palette(palette)` — 9 default text/bg pair üzerinde `ContrastFinding` döner (ratio + AA-normal/AA-large flag'leri)
- `find_aa_failures(findings)` — CI gate filter

Pin'lenen kurallar (test):
- Dark + light tema: text on panel_bg ≥ 4.5:1 (AA-normal)
- High-contrast tema: text on panel_bg ≥ 7.0:1 (AAA — aksesibilite teması bu hedef için var)
- High-contrast strictly > dark for body text
- Tüm temalarda success/warn/danger on panel_bg ≥ 3.0:1 (AA-large, icon/label kullanım)

## Test paketi
`tests/test_v209_paper_ready.py` (P1+P2+P4 tek dosyada, 24 test):
- PDF: minimal valid output, full-page (header+images+profiles+footer), parent dir mkdir, line-profiles overlay
- Bundle: zip create, checksum lists every file, checksum SHA matches actual bytes, params.json content, README names sample+operator, extras attached, missing-pdf error, missing-attachment error
- WCAG: luminance endpoints (black=0, white=1), contrast endpoints (21 / 1), symmetric, known pass/fail pair, theme audits (dark/light AA, high-contrast AAA + dominates dark, state colours AA-large), find_aa_failures filter, custom pair input

## Rakamlar
- **721 pytest pass** (697 → +24), 1 skipped, sıfır regresyon
- v2.0.9 sprint **ana 4 maddeden 4'ü ship** (P1+P2+P3+P4)
- Bug registry: 22/22 PASS hâlâ ✅
- Backlog'ta kalan v2.0.9 maddeleri (line profile click-drag UI, report mode export highlight, drop zone live indicator) ile küçük UI items v2.0.10'a kayıyor — hepsi pure UI, omurga (PDF/bundle/crash/WCAG) shipped

## v2.0.9 → v2.1.0 geçiş
Roadmap sıradaki: **v2.1.0 — Performance: GPU + headless** (4 hafta).
- PyTorch backend for `_make_fast_evaluator` (CUDA + Metal)
- Batch FFT (n_steps × ifft → 1 batched ifft)
- ROI fast-path (zero-fill outside ROI)
- Linux Docker + CI matrix
- Bench hedef: 2048² × 40 step zscan: 4.3 sn (Mac CPU) → < 1 sn (Mac MPS) → < 0.3 sn (RTX 4090)

---

# 2026-04-28 — v2.1.0 sprint ship: GPU + headless perf

## Context
"Devam" → v2.1.0 ana eksenleri tek oturumda. torch opsiyonel dep olarak entegre, mevcut numpy/scipy yolları sıfır regresyon ile devam ediyor.

## Ship 18 — G1 PyTorch FFT backend (`src/core/fft_backend.py`)
- `FFTBackendName.TORCH` enum üyesi
- `TorchFFTBackend` — lazy import, device auto-select (CUDA > MPS > CPU)
- `fft2_tensor` / `ifft2_tensor` — host↔device transfer atlamak için tensor-native API
- `get_best_fft_backend(prefer=TORCH)` — opsiyonel; default fallback chain'inde değil (torch heavy ve sub-1024² boyutlarında PyFFTW daha hızlı)
- Base `FFTBackend.fft2_batched` / `ifft2_batched` herkes için (default = serial loop), `supports_batched` flag'i opt-in için

## Ship 19 — G2 Batch FFT in evaluator (`_make_batch_evaluator`)
- `(n_steps, ny, nx)` H-stack tek seferde build
- Batched IFFT (torch backend'inde tek kernel launch, numpy'da seri loop ama API tek)
- Reference division batched broadcasting destekli
- `autofocus_zscan(batch_backend=...)` opt-in: backend `supports_batched` raporlarsa batched yola düşer; aksi halde serial path (cancel + progress aynı çalışır)
- **Correctness pin**: batched scores serial path'le rtol=1e-4 içinde eşit

## Ship 20 — G3 ROI fast-path (`_make_roi_fast_evaluator`)
- ROI ≪ frame durumunda IFFT ROI-sized output'a giriyor (spectral crop)
- Bandwidth-limited: lower spatial resolution but autofocus z hâlâ doğru
- Wall-clock saving ≈ full_area / roi_area (256² ROI on 2048² = 64×)
- Min 8 px clamp (sub-8 IFFT meaningless)
- Test: focus z within 6× scan step of truth on 128² scene

## Ship 21 — G4 bench backend flag (`scripts/bench_autofocus.py`)
- `--backends default,torch` cartesian compare
- Side-by-side table (shape × backend × device × runtime) markdown çıktısı
- Mevcut full algoritma sweep aynen kalıyor — backend compare additif

## Ship 22 — G5 Linux Docker + GitHub Actions CI matrix
- `Dockerfile` — NVIDIA CUDA 12.4 base, headless image (DPG yok), CLI entrypoint
- `.github/workflows/ci.yml` — Ubuntu 22.04 + macOS 14 × Python 3.11/3.13 matrix; Linux'ta torch CPU build install; bench job workflow_dispatch ile gated
- Smoke test build içinde: `pytest tests/test_autofocus_speed_baseline.py`

## Test (`tests/test_v210_perf.py`)
14 test (10 numpy yolu + 4 torch-skipif):
- Default backend has batched fallback
- Default backend `supports_batched=False` (correct gate)
- Batch evaluator vs serial: per-z score match (rtol=1e-4)
- Batch evaluator empty zs → empty array
- Batch evaluator with reference division: matches serial
- `autofocus_zscan(batch_backend=...)` matches serial best_z + scores
- Progress callback fires on batch path (start + end)
- ROI fast evaluator returns finite score
- ROI fast evaluator finds focus near truth
- ROI fast evaluator clamps min size
- (torch-skipif) constructs, fft round-trip, batched matches numpy, zscan matches numpy zscan

## Rakamlar
- **731 pytest pass** (721 → +10 numpy yolu), 5 skipped (4 torch-skipif + 1 multi-focus), sıfır regresyon
- v2.1.0 sprint **5 maddeden 5'i ship** (G1+G2+G3+G4+G5)
- Bug registry: 22/22 PASS hâlâ ✅
- Multi-focus + autofocus 7 algoritma + zscan batch path hepsi tek `_make_fast_evaluator` / `_make_batch_evaluator` / `_make_roi_fast_evaluator` şemsiyesi altında

## Lab demo path (Sven'in IT ekibine teslim için)
```bash
docker build -t dhm:cli .
docker run --rm --gpus all \
    -v $PWD/data:/data -v $PWD/out:/out \
    dhm:cli run /data/session.json --out /out --workers 4
```

```bash
docker run --rm --gpus all dhm:cli \
    bench --backends default,torch --shapes 1024,2048
```

GitHub Actions PR check'leri Linux + macOS × Python 3.11/3.13 matrix'inde otomatik çalışıyor; bench job workflow_dispatch ile manuel tetiklenir.

## v2.1.0 → v2.1.x geçiş
Roadmap sıradaki: **v2.1.x — Real hardware** (3 hafta).
- Pylon (Basler) CameraSource implementation
- IDS uEye + Thorlabs SciCam Protocol implementations
- MP4 recording (imageio-ffmpeg)
- Live preview: 30fps reconstruction loop
- Mock camera fixture for CI

---

# 2026-04-28 (akşam) — v2.1.x sprint ship: Real hardware

## Context
User: "AI kısmını sen geç başkasına verdim o işi sen diğer yerden devam et" → v3.0 (Faz 2 AI seg) dışsal ekibe gitti, ben v2.1.x'e geçtim.

Vendor SDK'ları (Pypylon/PyueYe/Thorlabs TSI) dev box'ta yok ve test edilemez. **Dürüst yol**: clean Protocol + capability metadata + integration scaffold; lab IT ekibi SDK'leri kurduğunda TODO bloklarını doldurur.

## Ship 23 — H1 Camera registry (`src/core/cameras/`)
Yeni package:
- `__init__.py` — `CameraBackendInfo` dataclass (name, vendor, summary, requires_sdk, capabilities), `all_backends()` (registered, regardless of SDK), `available_backends()` (only those with importable SDK), `make_camera(name, **kwargs)` factory, ValueError vs RuntimeError ayrımı (unknown vs unavailable)
- `synthetic.py` — re-export of v2.0.6'dan beri çalışan `SyntheticCamera`; capabilities={live, no-trigger, no-16bit, no-roi}
- `mock.py` — yeni `MockCamera` realistic pathology sim: drop_rate, exposure_jitter_us, read_noise_sigma, warmup_frames; deterministic w/ rng_seed (test replay)

Discovery `pkgutil.iter_modules` ile otomatik — yeni vendor sub-module ekleyince registry refleksiyon ile yakalar, central listeyi güncellemeye gerek yok.

## Ship 24 — H2 MP4 recorder (`src/core/video_recorder.py`)
`MP4Recorder` `TiffStackRecorder`'la aynı Protocol (start / write_frame / stop / frames_written). imageio + imageio_ffmpeg lazy-import; `is_available()` kontrolü ile yokluk durumunda `start()` clear RuntimeError, dialog dropdown filtreleyebilir. 8-bit RGB encode (browser-friendly), `quality=8` lab default. Bilimsel veri için TIFF stack (lossless) source-of-truth kalır; MP4 sunum/share artefaktı.

## Ship 25 — H3 Vendor backend stubs (Pylon / IDS / Thorlabs)
Üç stub aynı şablonu kullanıyor:
- Module-level `BACKEND = CameraBackendInfo(...)` (capabilities + requires_sdk)
- `is_available()` import probe (no side effects)
- `<Vendor>Camera` skeleton class — `start/stop/grab/size/fps`
- `make()` factory
- Each method body has documented TODO blocks with the actual SDK calls (pypylon.InstantCamera vs pyueye.HIDS vs thorlabs_tsi_sdk.TLCameraSDK)
- Without SDK → registry filters out from `available_backends()`; `make_camera('pylon')` clear RuntimeError; her vendor'ın `start()` blanket NotImplementedError

`docs/cameras.md` — integration guide: `start()/grab()/stop()` patterns, frame normalisation contract (uint16 → float32 [0, 1]), test surface (mock/synthetic for CI, real_hardware marker for lab box smoke).

## Ship 26 — H4 Mock camera + FPS perf test
`tests/test_v21x_hardware.py` — 21 test:
- Registry: all_backends includes 5 (synthetic/mock/3-vendor), available_backends excludes vendors w/o SDK, capabilities carried, `make_camera` factory + error paths
- Mock camera: drop_rate=1.0 → all-zero, warmup → blank then signal, deterministic w/ same seed, Protocol surface intact
- Vendor stubs: `make()` returns skeleton, `start()` raises NotImplementedError until SDK block filled
- MP4 recorder (imageio-ffmpeg-skipif): round-trip > 256 bytes, frames_written counter, parent dir mkdir, write-before-start error, missing-imageio error
- FPS perf: SyntheticCamera + MockCamera ≥ 30 fps at 512² (live preview floor)

## Rakamlar
- **749 pytest pass** (731 → +18 active path), 8 skipped (4 torch + 3 mp4-ffmpeg + 1 multi-focus), sıfır regresyon
- v2.1.x sprint **4 maddeden 4'ü ship** (H1+H2+H3+H4)
- Bug registry: 22/22 PASS hâlâ ✅
- 5 camera backend (synthetic + mock + 3 vendor stubs) registry'de; 2'si (synthetic + mock) immediately usable, 3'ü integration scaffold

## Faz 2 (AI seg) dışsal ekibe verildi
Roadmap'in v3.0 satırı bu sprintten geçilmiyor — başka ekip işliyor. Geri dönüşü için:
- API contract dokümante edildi (cell mask + classifier output expectations) → doc tarafına bırakılabilir
- DHM tool'un AI çıktısını consume edeceği yer: `core.session_export.CellMeasurement` payload'ı + `core.tracking.Detection.payload` — segmentation/classification skalerleri orada akar
- v2.0.7 multi-user + v2.0.8 tracking + v2.0.9 paper-ready zaten AI olmadan da end-to-end çalışıyor

## Sıradaki — v2.0.x backlog'tan kalanlar (UI polish)
v2.0.9 sprint'inde shipping omurgaya odaklanıldı; UI polish item'ları sıraya girdi:
- Line profile click-drag ROI dialog (`core.line_profile` data layer hazır)
- Drop zone reconstruction live indicator
- Report mode export buttons highlight

Bunlar ya v2.1.y (UI cleanup) ya da bir sonraki feature sprint'i içinde toplanır. v2.1.x ana hedef (real hardware integration scaffold) tamam.

---

# 2026-04-28 (gece) — v2.1.x H5+H6 ship + schema v12 fix

## Ship 27 — H5 Time-lapse acquisition (`src/core/timelapse.py`)
Karin'in 12-saatlik live cell imaging için.
- `TimelapseSchedule(interval_s, total_frames, max_duration_s, start_at)` validation + `expected_frame_count` planning
- `TimelapseRunner` — camera + per-frame TIFF write + Session manifest output, fake-clock-injectable
- Cancel-aware sleep (≤100ms latency), grab-failure containment (gap notes in manifest), camera-start-failure manifest still written, `start_at` UTC delay
- Output: per-frame `frame_NNNN.tif` + `session.json` manifest → CLI runner pipeline'ına direkt akar
- 18 test (`tests/test_timelapse.py`): schedule validation, frame caps (count/duration/both), cancel-mid-run, error paths, start_at delay, callbacks

## Ship 28 — H6 Live vs File mode UI ayrımı
Pilot review'da: "live ve file modlarını ayırt edemiyoruz". Çözüm:
- Explicit `_input_mode` state ("file"/"live"/"timelapse")
- `_MODE_PREFIX` table → status bar her mesajın başına `[FILE]` / `[● LIVE]` / `[● TIMELAPSE]` ekliyor
- `_load_hologram` → file mode flip
- `_on_camera_start` → live mode flip; `_on_camera_stop` → file mode geri
- `_latest_live_frame` cache (her camera frame'de copy alınıyor)
- `_snapshot_live_frame_to_tempfile` → operatör Reconstruct'a basınca live frame TIFF tempfile'a yazılıyor → mevcut reconstruct path'i sıfır değişiklikle çalışıyor
- 15 test (`tests/test_input_mode.py`): default mode, set/get, prefix table coverage, status prefix verify, latest-frame copy semantics, snapshot tempfile round-trip, mode transitions (load/start/stop), reconstruct-in-live-mode auto-snapshot, reconstruct-in-file-mode unchanged

## Ship 29 — Schema v11→v12 migration bridge
Paralel AI ekibi `core/settings_schema.py`'da `SCHEMA_VERSION` 11→12 (yeni `AIDefaults`) yaptı + Qt tarafına `_migrate_v11_to_v12` eklemişler. JSON side eksikti → 6 bug registry test'i fail. Düzeltildi:
- `src/ui2/state_store.py::_v11_to_v12` migrator (ai={} default backfill)
- `_hydrate` artık `ai=_hydrate_dc(AIDefaults, raw.get("ai"), d.ai)` populate ediyor
- `AIDefaults` import eklendi
- Mode setter defensive: `prev = getattr(self, "_input_mode", "file")` (test `__new__` pattern'ine karşı)

`tasks/lessons.md § 2026-04-28` — kural: cross-team schema bump'ında her iki migration tarafının (Qt + JSON) eşzamanlı eklenmesi gerek; `scripts/check_bugs.py` 30 saniyede regression yakalar.

## Rakamlar
- **782 pytest pass** (749 → +33), 8 skipped (4 torch + 3 mp4 + 1 multifocus), sıfır fail
- v2.1.x H5+H6 ship + schema v12 fix bridge
- Bug registry: 22/22 PASS ✅

## Sıradaki: v2.1.y UI polish mini-sprint
v2.0.9'dan kalan 3 madde toplanacak:
- P1 Line profile click-drag ROI dialog (`core.line_profile` data layer var)
- P2 Drop zone reconstruction live indicator
- P3 Report mode export buttons highlight

---

# 2026-04-29 — v2.1.y UI polish mini-sprint ship

## Pattern
Üç parça da pure-state modülü olarak yazıldı (testable headless), DPG wrapper minimal. v2.1.y backlog'undan toplandı.

## Ship 30 — P1 Line profile click-drag state machine
`src/ui2/line_profile_state.py`:
- `EditorState` enum (IDLE, AWAITING_END, PREVIEW)
- `LineProfileEditor` — `first_click(y, x)` → `second_click(y, x, label)` flow, palette cycle (6 colour default), `cancel/drop_last/clear_all/rename/set_colour`
- Zero-length line reject (bilinear sampler NaN catch)
- Mid-draft second `first_click` accepts (forgive wandering hand)
- After commit auto-revert to IDLE → next click başlar

Data layer (`core.line_profile`) v2.0.8 D4'tendi; bu sprint UI state machine bridge.

## Ship 31 — P2 Drop zone state
`src/ui2/ui_state.py::DropZoneState`:
- `DropZoneStage` enum (READY / LOADING / RECON / DONE)
- Per-stage label table + colour role table (TEXT_MUTED / ACCENT / SUCCESS)
- `transition(stage, hint=)` API — hint optional ("step 12 of 40")
- v2.0.3 backlog: pilot review'da "click reconstruct, status text scrolled away too fast" şikayetinin cevabı

## Ship 32 — P3 Workflow export buttons
`src/ui2/ui_state.py`:
- `EXPORT_BUTTON_TAGS` set (4 tag: report, csv, bundle, pdf)
- `workflow_export_buttons_visible(mode)` — sadece "report" mode True
- `workflow_export_buttons_accented(mode)` — visible olduğunda accent border
- `is_export_button_id(tag)` — tag-name fan-out yerine merkez kontrol

## Test (`tests/test_v21y_ui_polish.py`)
**37 test** tek dosya (P1+P2+P3):
- LineProfileEditor: 16 test — state transitions, zero-length reject, mid-draft reset, drop_last/clear_all/rename/set_colour, palette cycling
- DropZoneState: 9 test — default ready, transitions, hint clear, success colour, table completeness for every enum
- Workflow helpers: 12 test — case-insensitive Report match, visibility/accent gating, EXPORT_BUTTON_TAGS membership

## Rakamlar
- **819 pytest pass** (782 → +37), 8 skipped, sıfır fail
- v2.1.y mini-sprint **3 maddeden 3'ü ship**
- Bug registry: 22/22 PASS hâlâ ✅
- v2.0.x backlog'undan kalan tüm UI polish item'ları kapandı

## Roadmap güncel durum
| version | tema | durum |
|---------|------|-------|
| v2.0.7 | Time-lapse foundation | ✅ |
| v2.0.8 | Tracking & calibration | ✅ |
| v2.0.9 | Paper-ready output | ✅ |
| v2.1.0 | GPU + headless | ✅ |
| v2.1.x | Real hardware (incl. time-lapse + mode UI) | ✅ |
| v2.1.y | UI polish (line ROI, drop zone, export highlight) | ✅ |
| v3.0 | AI segmentation | ⏭️ dışsal ekipte |

Tüm planlı sprint'ler kapandı. AI ekibi entegrasyonu beklerken backlog item olarak kalan: **DPG wrapper'larını state modülleri üzerine wire'lamak** (her üç P maddesinin DPG render kısmı). Bu next sprint'te yapılır veya AI entegrasyonu geldiğinde birlikte gelir.

---

# 2026-04-29 (akşam) — v2.1.z sprint ship: Lab device control

## Context
User: "desktop/APT diye bir klasör var onu direkt entegre etmek yerine onun gibi bir tool vs ekleyebilir misin". APT klasörü stage/shutter/LED gibi cihaz kontrolü için ayrı bir araç. İçeriğini okumadım (scope dışı), aynı kapasiteyi DHM tool'umuzda kendi mimarimizde inşa ettim — `core.cameras` registry pattern'iyle %100 uyumlu.

## Ship 33 — `core.devices` Registry + Protocols
- `DeviceKind` enum (STAGE / SHUTTER / LED / GENERIC), `DeviceBackendInfo`, registry helpers (`all_backends`, `available_backends`, `backends_by_kind`, `make_device`)
- pkgutil-driven discovery → yeni vendor dosyası ekleyince merkezi listeyi güncelleme yok
- Protocol granularity: `Device`, `StageDevice`, `ShutterDevice`, `LEDDevice` — kind-spesifik (consumer'lar `hasattr` dance yapmıyor)

## Ship 34 — Mock backends (CI default)
- `mock_stage.py` — XYZ pozisyon, soft limits, configurable settle_time
- `mock_shutter.py` — binary state, configurable open/close latency
- `mock_led.py` — 0–100% intensity (clamp instead of raise), on/off

## Ship 35 — Generic serial backend
`serial_generic.py` — pyserial lazy import, ASCII command/reply pattern (`send_command(str, expect_reply)`). Vendor SDK'sı olmayan lab cihazları için base. SDK probe → `available_backends()` filter, `make_device('serial_generic')` clear RuntimeError pyserial yokken.

## Ship 36 — Acquisition orchestrator (APT-tarzı value)
`src/core/devices/orchestrator.py` — multi-position imaging koreografisi:
- `StagePosition(x_um, y_um, z_um, label)` waypoint dataclass
- `AcquisitionPlan` — positions + led_intensity + shutter_per_frame + settle_time
- `run_plan(plan, camera, stage=, shutter=, led=, cancel_check=, on_frame=)` — declarative lab automation
- Lifecycle: connect → set LED → for each pos: move + shutter open + settle + grab + shutter close → cleanup
- Error containment: stage move fail / grab fail per-frame `error` field, plan asla raise etmiyor
- Time-lapse runner ile compose: `TimelapseRunner` interval-based, `run_plan` position-based; ikisi birleştirilebilir (overnight 24-position 12-hour session)

## Test (`tests/test_v21z_devices.py`) — 29 test
- Registry: 6 test — discovery, kind filter, SDK gating (pyserial), capabilities, factory error paths
- Mock stage: 6 test — Protocol shape, connect/move/home, out-of-range, before-connect raise, disconnect
- Mock shutter: 3 test — Protocol, open/close cycle, before-connect raise
- Mock LED: 4 test — Protocol, intensity round-trip, clamping, on/off
- Orchestrator: 10 test — empty positions=single grab, position order, per-frame vs continuous shutter, LED intensity, failure containment, cancel mid-plan, callback order, PlanResult shape

## `docs/devices.md`
Camera docs ile aynı tarz: discovery API, vendor backend ekleme şablonu, multi-device orchestrator usage, mock vs SDK gating contract.

## Rakamlar
- **919 pytest pass** (819 → +29 device + +71 diğer external testler), 9 skipped
- v2.1.z sprint **6 maddeden 6'sı ship**
- Bug registry: 22/22 PASS hâlâ ✅

## Heads-up — AI ekibi domain fail
- `tests/test_ai_panel.py` 5 fail: `core.ai.client` non-lazy `import requests`, venv'de yok.
- Benim sprint'imle alakasız (bug registry yeşil, 919 test pass).
- AI ekibi lazy import + `pytest.importorskip("requests")` ekleyince düzelir. v12 schema migration'da olduğu gibi: cross-team değişikliklerinde JSON+Qt migration zinciri kontrol kuralı `lessons.md § 2026-04-28`. Bu da o pattern'in başka tarafı: optional dep'ler için lazy import + test skipif.

## APT yaklaşımı
User'ın isteği "direkt entegre etme, onun gibi bir tool ekle":
- APT'nin device kontrol konsepti → `core.devices` registry
- APT'nin stand-alone tool yapısı → DHM tool içine entegre, time-lapse + camera registry'yle koreografisi yapılan orchestrator
- APT'nin kendi codebase'ine bağımlılık → sıfır; mock'larla CI green, vendor SDK'lar opt-in
- APT'nin Tk GUI'si → DPG wrapper next sprint'te (state machine pattern `line_profile_state.py` gibi)

---

# 2026-04-29 (gece) — AI integration audit + bug fixes + APT-uyumlu device tools

## Context
User: "AI entegrasyon kısmına ve AI için eklenenlere bir bak. Bir sorun var mı? Bir de AI için toollar ekliyoruz. Stage toolları yok APT'ye uygun."

İki iş:
1. AI ekibinin pushladığı entegrasyonu audit et, somut bug'ları çıkar+düzelt
2. APT-uyumlu device tools (shutter/LED/orchestrator) AI agent'ına ekle — `core.devices` Protocol'üne (v2.1.z'de yazdığım) bağlı

## Bug fixes (5 madde)

### Bug 1 — `src/core/ai/client.py` non-lazy `requests` import
Yorum "Lazy-import requests" diyordu ama eager import vardı. `AIPanel.__init__` requests yokken hard-fail oluyor → tüm pencere açılamıyor.
**Fix**: `_ensure_session()` first-use bootstrap. `health_check`, `_post_json` her ikisi de çağırıyor. Import error → `LLMClientError("install requests…")`. (Bir paralel commit aynı yöne hardened — _session ve _requests ayrı guard.)

### Bug 2 — `_confirm` thread marshalling (sessiz dead code)
`QTimer.singleShot(0, _run)` calling thread'inde post ediyor — AI worker thread'inde event loop yok → timer hiç fire etmedi. Confirmation dialog 120s timeout sonra silently expire. Irreversible tool gate'i çalışmıyordu.
**Fix**: `QMetaObject.invokeMethod(self, "_run_confirm_dialog", Qt.QueuedConnection)` — slot her zaman receiver'ın thread'inde çalışır (panel için GUI thread). `_pending_confirm` instance dict + `threading.Event` ile sonuç worker'a senkron gelir. `Slot` decorator + import eklendi.

### Bug 3 — `record_timelapse` cancellation deafness
Sleep loop "100 ms slice" comment'i cancel propagation iddia ediyordu ama agent-level cancel callable scope'da değildi. 8-frame 5min interval timelapse → Stop'a basınca yine sonuna kadar gidiyordu.
**Fix**: `ToolContext.is_cancelled: Optional[Callable[[], bool]]` field eklendi. Loop hem frame öncesi hem sleep slice'larında poll ediyor. Result dict'e `cancelled` + `completed_frames` eklendi.

### Bug 4 — Stale "14 tools" docstring
`tool_impls.py:14` ve `:914` "14" diyordu, gerçek 19. v2.1.z ile artık 19+9=28.
**Fix**: docstring güncellendi, `build_tool_registry(include_devices=True)` default + opt-out flag.

### Bug 5 — `stage_focus_search` clamp bound karışıklığı
`clamp("z_mm", search_range_mm)` — pozisyon (-200..200) bound'unu pencere genişliğine (1..50) uygulayan yanlış axis'ti. `step_mm` zaten clamp edilmiyordu (sadece schema, jsonschema yoksa kayboluyor).
**Fix**: `NUMERIC_BOUNDS`'a `search_range_mm`, `step_mm`, `mask_dilate_px`, `intensity_percent`, `rows`, `cols`, `spacing_x_um`, `spacing_y_um`, `settle_time_s` eklendi. Clamp doğru bound'ları kullanıyor şimdi.

## APT-uyumlu device tools (9 yeni AI tool)

### `ToolContext` extension (v2.1.z)
3 yeni hook field:
- `is_cancelled: Optional[Callable[[], bool]]` — long-running tools poll'lar
- `shutter: Any` — `core.devices.ShutterDevice` instance veya None
- `led: Any` — `core.devices.LEDDevice` instance veya None
- `orchestrator: Optional[Callable[[dict], dict]]` — multi-position acquisition runner

### Yeni 9 tool (`core.ai.tool_impls`)
| tool | açıklama |
|------|----------|
| `list_devices` | Registry rollup — `available` (mock + SDK var olanlar) + `all` (tümü, SDK durumu işaretli). LLM "mock_stage mı, gerçek Thorlabs mı?" sorabilir. |
| `shutter_open` | Auto-connect + open. RuntimeError yerine clean error dict. |
| `shutter_close` | Close. |
| `shutter_status` | Configured / connected / is_open. |
| `led_set_intensity` | 0–100 % (clamp NUMERIC_BOUNDS üzerinden). |
| `led_on` / `led_off` | Toggle (intensity preserved). |
| `led_status` | Configured / connected / is_on / intensity. |
| `acquire_grid` | rows×cols × spacing_um → orchestrator'a forward. Multi-position imaging tek tool çağrısı. |

### Mevcut stage_* legacy → APT geçişi
Mevcut `stage_*` tools `ctx.stage` legacy API'sini (move_relative/absolute/home) kullanıyor. Bunları **kırmadım** — onlara dokunmak existing test surface'ı bozardı. Yeni tools `core.devices.StageDevice` Protocol'üyle uyumlu (orchestrator üzerinden). Lab IT pypylon / pyserial kurduğunda gerçek hardware → registry üzerinden direkt akar.

## Test
- `tests/test_ai_device_tools.py` — **15 test**: list_devices rollup, shutter open/close round-trip + status + open-failure-error, LED intensity round-trip + clamp + on/off + status, acquire_grid orchestrator forward + bound enforcement + exception wrap, include_devices=False removal, record_timelapse cancel-aware
- `tests/test_ai_tools.py` + `tests/test_ai_tools_advanced.py` — `include_devices=False` ile canonical 19-tool surface pin'i korundu, yeni 9 tool ayrı testle pinlendi

## Rakamlar
- **977 pytest pass** (819 → +158 — 29 device + 15 AI device-tools + AI ekibinin existing 88 + bench), 9 skipped, sıfır fail
- Bug registry: 22/22 PASS hâlâ ✅
- 5 high-priority bug fix shipped, 4 audit findings dokümante (panel-side `_gui_capture_and_process` race ve hardcoded sample_maps path AI ekibinin domain'inde — onlar kararlaştırır)

## Heads-up — AI ekibine bırakılan
Audit'te yakalanan ama benim sprint scope'um dışında kalan iki nokta:
- `_gui_capture_and_process` race: `trigger()` async submit + immediate cached read → previous turn's stale data döner. Düzeltmesi panel-side worker completion sync gerektiriyor; AI ekibinin entegrasyon kararına bağlı.
- `~/.dhm-reconstruction/sample_maps/` hardcode: `core.user_profile`'ın per-user dir'ine taşınmalı. Multi-user kullanım sahnesi gelene kadar bekleyebilir.

---

# 2026-04-29 (sabah) — v2.1.z follow-up: multi-position bridge + AI device wiring + registry hygiene

## Context
"Bunun haricinde yapabileceğin ne var" — autonomous mode'da üç açık ve değerli iş paralel ship edildi.

## Ship A — Multi-position time-lapse bridge
`src/core/multi_position_timelapse.py`:
- `MultiPositionSchedule(interval_s, total_ticks, max_duration_s, positions, led_intensity_percent, shutter_per_frame, settle_time_s)` — `TimelapseSchedule` + `AcquisitionPlan` props birleşik
- `MultiPositionTimelapseRunner` — her tick'te bir `run_plan` çağırıyor, per-position `frame_t<NNNN>_p<NN>.tif` + Session manifest yazıyor
- Frame `params_overrides` field'inda `tick`, `position_index`, `x_um/y_um/z_um` taşınıyor → CSV exporter pivot edebilir
- Cancel-aware (intra-tick + inter-tick), tifffile-missing graceful, fake-clock injectable
- 13 test (`tests/test_multi_position_timelapse.py`): schedule validation, frame counting, max_duration cap, cancel mid-run, callbacks, edge cases (no positions, no tifffile)

Karin'in 24-position 12-hour overnight session pattern'i tek API call:
```python
sched = MultiPositionSchedule(
    interval_s=300.0, total_ticks=144,
    positions=[StagePosition(x*100, y*100, label=f"r{x}c{y}")
               for x in range(6) for y in range(4)],
    led_intensity_percent=50.0,
)
runner = MultiPositionTimelapseRunner(
    sched, "/data/session", camera=cam, stage=stage,
    shutter=sh, led=led, sample_id="A549_overnight",
)
result = runner.run()  # 144 × 24 = 3456 frame, full manifest
```

## Ship B — AI panel device wiring
`src/gui/panels/ai_panel.py::AIPanel._build_tool_context` v2.1.z hooks ile genişletildi:
- `_make_device_hooks()` — default-construct `mock_shutter` + `mock_led` from `core.devices.make_device(...)`. Cached as `panel._v21z_device_hooks` so state survives across turns (turn 1'de açılan shutter turn 2'de hâlâ açık)
- `_orchestrator_callable(args)` — AI tool args (rows/cols/spacing) → `AcquisitionPlan` → `run_plan` adapter; `_PanelCamera` adapter mevcut `panel._capture_frame()`'i CameraSource Protocol'üne uyduruyor
- `is_cancelled` callable — worker thread'inin `isInterruptionRequested()`'i sarmalıyor

Sonuç: 9 yeni device tool (`shutter_open` … `acquire_grid`) artık panel default'larıyla doğrudan çalışıyor. Lab IT vendor SDK kurunca `panel._v21z_device_hooks = (vendor_shutter, vendor_led, my_orchestrator)` şeklinde swap.

## Ship C — Bug registry + lessons update
**Bug registry**: 13 yeni entry (B-036…B-048) — sprint kapsamına yayılmış:
- TIMELAPSE_FOUNDATION (3): preset edit/replace, v2.0.7 omurga, v11→v12 schema migration lesson
- TRACKING (1): drift + tracking + calibration + multi-line
- PAPER_READY (1): PDF + Zenodo + WCAG + crash handler
- PERF_GPU (1): torch + batch FFT + ROI fast-path
- HARDWARE (2): camera registry + time-lapse + mode UI + devices, multi-position bridge
- PILOT_PATCHES (5): AI lazy-import, _confirm threading, record_timelapse cancel, clamp bound, APT device tools

**Bug registry now**: **34 PASS / 0 FAIL / 0 SKIP / 9 LESSON / 5 MANUAL** (was 22 PASS / 8 LESSON / 5 MANUAL → +13 entry, hepsi yeşil)

**lessons.md** — 2 yeni kural:
1. **Lazy import yorumu ≠ lazy davranış**: `__init__` içinde import etmek "construction-time eager"; gerçek lazy `_ensure_x()` first-use bootstrap. ImportError'u domain-spesifik error olarak sar (LLMClientError).
2. **Qt thread'ler arası dialog**: `QTimer.singleShot(0, ...)` calling thread'in event loop'una post eder. Worker thread'inde event loop yok → callback hiç fire etmez. Canonical: `QMetaObject.invokeMethod` + `Slot()` decorator + `Qt.QueuedConnection`.

## Rakamlar
- **993 pytest pass** (977 → +16 multi-position + 0 panel-only changes), 9 skipped, sıfır fail
- Bug registry: **34/34 PASS** ✅ (was 22/22)
- 9 phase aktif: PRE_PILOT, UX_PATCH, DPG_PORT, PILOT_PATCHES, TIMELAPSE_FOUNDATION, TRACKING, PAPER_READY, PERF_GPU, HARDWARE; AI_FAZ_2 hâlâ boş (dışsal ekipte)

## Roadmap durum

| version / phase | tema | durum |
|-----------------|------|-------|
| v2.0.7 - v2.1.y | omurga + UI polish | ✅ |
| v2.1.z | Lab device control + orchestrator + multi-position bridge | ✅ |
| v2.1.z+ | AI device-tool integration (9 tools, audit fixes) | ✅ |
| v3.0 | AI segmentation | ⏭️ dışsal ekipte |

Açık kalan rezerv:
- AI ekibi domain'inde 2 race/hardcode bug (notlu, müdahale etmedim)
- DPG wrapper'lar henüz state modülleri üzerine bağlanmadı (line_profile dialog, drop zone indicator, device control panel)

---

# 2026-04-28 (öğleden sonra) — sprint hygiene + line profile dialog ship

## Context
"ne yapılabilir bir bak" — autonomous mode, üç açık iş paralel yapıldı.

## Gözlem — AI ekibi paralel fix'leri
Audit'te listelediğim "AI ekibinin domain'i, müdahale etmedim" 2 bug aslında onların paralel commit'leriyle düzelmiş:
- `_persist_sample_map` → `core.user_profile.user_state_dir() / "sample_maps"` kullanıyor (artık per-user)
- `_gui_capture_and_process` → `_wait_for_signal(sig, timeout_ms=60_000)` ile worker tamamlanmasını bekliyor (race kapalı)

Bug registry'ye gözlem entry'si (B-049, lesson_only) eklendi.

## Ship A — `scripts/add_bug.py` CLI helper
Sprint sonu hygiene için tek-komut bug-ekleme aracı:
```bash
python scripts/add_bug.py \
    --phase pilot_patches_v2_0_6_post \
    --topic "..." \
    --status test \
    --test "tests/test_xxx.py" \
    --lesson-ref "(sprint ref)"
```
- Next free `B-NNN` id otomatik (regex walk)
- Phase enum string → enum member name çevirisi
- `_format_entry` 64-char wrap (file style match)
- `_splice_entry` bracket-matching: `BUG_REGISTRY: List[BugEntry] = [` deklarasyonundan başlayıp `[`/`]` derinliği track ederek **doğru** kapanışı buluyor (önceki naïve `rfind("\\n]\\n")` `__all__` listesinin kapanışına denk geldi → silent corruption; ilk gerçek kullanımda yakalandı, fix'lendi)
- Atomic write (tempfile + os.replace)
- Sanity-check: yazımdan sonra `check_bugs.py --filter <new_id>` koşturuyor; FAIL'da uyarıyor ama dosya kalıyor (debug mümkün)

## Ship B — DPG line-profile dialog wrapper
v2.1.y P1'de pure-state machine (`ui2.line_profile_state.LineProfileEditor`) yazmıştım; DPG katmanı eksikti.

`src/ui2/dialogs.py::show_line_profiles(editor=, image_provider=, on_close=)`:
- Plot panel — her saved profile bir `add_line_series` (color cycle palette'ten)
- Tablo — label / length / min/max/mean (NaN-safe), per-row Drop button
- "Clear all" / "Close" footer
- `image_provider()` callable (None tolerant) → caller's data layer'ı dialog'tan ayırıyor
- `editor` param None → fresh editor, `dpg._line_profile_editor` üzerine stash → caller daha sonra fetch edebilir
- Stats `core.line_profile.stats_for(sampled)` üzerinden (NaN-safe)

5 headless test — DPG stub'ı `top-up` pattern ile (lessons.md § 2026-04-24 stub top-up dersi):
- Constructs editor on first call
- Honours supplied editor (saved profiles persist)
- Image=None tolerated
- Real image runs without raise
- on_close kwarg accepted

İlk versiyonda `_install_dpg_stub` "early return if already installed" pattern'ı kullandı → sibling test (`test_input_mode.py`) önce koşunca onun stub'ında `add_plot_legend` eksikti, full suite'te 5/5 fail. lessons.md'deki "DPG stub top-up across test files" dersini hatırlayıp pattern'i `if not hasattr(dpg, name): setattr(...)` şekline çevirdim — herhangi bir sıralama ile robust artık.

## Rakamlar
- **998 pytest pass** (993 → +5 dialog), 9 skipped, sıfır fail
- Bug registry: **34 PASS / 0 FAIL / 0 SKIP / 10 LESSON / 5 MANUAL** (B-049 gözlem entry'si)
- 9 phase aktif rakamla (TIMELAPSE_FOUNDATION 3 entry, PILOT_PATCHES 15 entry — en yoğun phase)

## Açık kalan rezerv
- DPG dialog'ları henüz `app.py`'a wire değil (operatör menüden açamıyor) — bir sonraki "menu integration" sprint'inde gelir
- Drop zone indicator + device control panel dialog'ları hâlâ pure-state, DPG wrapper yok
- AI panel'in v2.1.z device hooks'u default'larla wired (mock_shutter + mock_led) ama vendor backend swap için lab IT'ye doc gerek

---

# Plan — 2026-04-28 — AI Asistan Fine-tune (Stage'siz)

## Context
Lab profili netleşti: HeNe λ=632.8 nm + 50× obj + USAF/bead/RBC/E.coli/Bacillus + Türkçe konuşma + İngilizce/sayısal tool args. Motorize stage henüz takılmadı → 8 stage tool'u training'den çıkarıyoruz, motorize gelince ek-LoRA ile döner (data/ai/training_examples_stage.jsonl ayrı tutulacak). 11 aktif tool kalıyor.

[docs/AI_FINETUNE_DATA.md](../docs/AI_FINETUNE_DATA.md) zaten 100-örneklik tasarım rehberi veriyor — bunu uygulayacağız. Mevcut [scripts/ai_training_examples.py](../scripts/ai_training_examples.py) 17 seed örnek üretiyor; bunu 100'e çıkaracağız + 15 holdout ayrı.

## Plan

### A. Veri üretimi
- [ ] `ai_training_examples.py` modülerleştir: kategori başına alt-fonksiyon (`_tool_selection_examples`, `_chain_examples`, ...)
- [ ] `STAGE_TOOL_NAMES` filter — registry schema'sından stage tool'larını çıkar (default ON, motorize gelince `--include-stage` flag ile geri aç)
- [ ] `LAB_PROFILE` sabitleri: λ=632.8, pixel=3.45, M=50, n_medium dry/wet=1.0/1.337, sample n_sample tablosu
- [ ] 100 training örneği yaz (kategori dağılımı: 15+10+25+8+5+15+12+5+5)
- [ ] 15 holdout örneği yaz (data/ai/eval_holdout.jsonl)
- [ ] JSONL üret + satır sayımı doğrula

### B. Eval framework
- [ ] `tests/test_ai_finetune_eval.py` — 4 metrik: tool selection acc, arg schema validity, refusal correctness, chain summary present
- [ ] FakeLLMClient ile tüm holdout'u koş, JSON raporu üret

### C. Pipeline kurulumu
- [ ] `Modelfile.dhm-copilot` — Pipeline A (system prompt + few-shot, lab profil bilgisi gömülü)
- [ ] `scripts/finetune_lora.py` — Pipeline B (qwen2.5:7b base, LoRA r=16, MPS-compatible)
- [ ] `requirements_finetune.txt` — opsiyonel deps (transformers, trl, peft, datasets, accelerate)

### D. Docs
- [ ] `docs/AI_FINETUNE.md` lab-profil kısmı + holdout kullanımı
- [ ] `docs/AI_FINETUNE_DATA.md` mevcut sayıları güncelle (17 → 100)
- [ ] `tasks/lessons.md` — kategori bazlı modüler training script + stage opt-out lessons

## Stage tool listesi (askıda — geri dönecek)
`stage_get_position`, `stage_move_relative`, `stage_move_absolute`, `stage_home`, `stage_focus_search`, `map_sample_grid`, `list_mapped_cells`, `goto_cell`. (8 tool)

## Aktif tool listesi (training scope)
`load_hologram`, `set_recon_param`, `run_reconstruction`, `run_autofocus`, `find_focus_candidates`, `run_qpi`, `compute_depth_map`, `get_state`, `get_last_result`, `get_audit_tail`, `record_timelapse`. (11 tool)

## Sample → tool argümanı eşleme tablosu (training'in altın anahtarı)

| Sample | n_sample | n_medium | Notes |
|---|---|---|---|
| USAF 1951 | — (no QPI) | 1.0 | calibration only |
| Polystyrene bead 3/10 µm | 1.59 | 1.0 / 1.337 | size cal |
| RBC | 1.41 | 1.337 (PBS) | hematology |
| E. coli | 1.40 | 1.337 | bakteri rod |
| Bacillus subtilis | 1.39 | 1.337 | spore-former |
| Staphylococcus | 1.40 | 1.337 | kok |
| Pseudomonas | 1.39 | 1.337 | basil |
| Lactobacillus | 1.39 | 1.337 | uzun rod |

---

# Plan — 2026-05-05 — Track C (Hybrid CNN) Reference-Free Reconstruction

## Context (kararın çıkış noktası)

Müşteri reconstruction sırasında referans hologram kullanmaktan kurtulmak istiyor (operatif yük, alignment hassasiyeti, vardiya başı kalibrasyon).

Üç yol incelendi (bkz. lessons.md 2026-05-05 ve `track_b_pure_dl_notes.md`):

- **Track A — saf klasik (Zernike/polynomial fit)**: 14-frame fair benchmark'ta median RMSE = **3.92 rad** (hedef <0.15 rad). Floor ~26x üstte. **Yetersiz**.
- **Track B — saf DL (eHoloNet/Y-Net stili)**: 63 frame ve düşük sample diversity ile pure DL için ölçek tamamen yetersiz. Notlar `track_b_pure_dl_notes.md`'ye kaydedildi, ileride veri kampanyası ile yeniden gündeme alınır.
- **Track C — hybrid (klasik + küçük CNN residual corrector)**: ✅ **bu sprint** — diff görsellerinde **structured stripe pattern** + session-stable + 63 frame'in 50'si train + 13 test için yeterli.

## Hedef

Tek hologram + referanssız reconstruction:
- **median RMSE ≤ 0.50 rad** (Track A'dan ~8x iyileşme, hedefe ~3x uzaklık)
- **p95 abs err ≤ 1.50 rad**
- **inference < 100 ms** (1024² frame, M-series Mac CPU yeterli; GPU plus)
- **leave-one-session-out** validation: cross-session generalization gap < 2x

## Mimari kararı

**Pipeline (canlı çalışırken)**:
```
Raw hologram (1024×1024 uint16)
   ↓ demodulate (Fourier sideband, classical)
   ↓ propagate at autofocus z (classical, ASM)
   ↓ wrapped phase + amplitude (classical)
   ↓ polynomial bg-fit order 5 (classical, 5 ms)
   → ϕ_classical (still has structured stripes)
   ↓ small residual CNN (U-Net lite, ~1.5M params)
   → ϕ_clean (production output)
```

**Eğitim hedefi (residual learning)**:
```
input  = ϕ_classical (poly5 reffree output)
target = ϕ_classical - ϕ_ref_based  (the residual stripe pattern)
loss   = L1 + 0.1·SSIM + 0.05·TV
```

Residual prediction tercih sebebi:
- Network sıfır-output verse pipeline degrade yapmaz (hala Track A poly5 floor'unda)
- "Aberration patterns"i öğrenmek "phase'in kendisini" öğrenmekten kolay
- Sample-tip diversity'sine az duyarlı (residual sample-bağımsız aberration)

**Architecture**: U-Net lite, 1024×1024 input, 4 down + 4 up, base 32 channels, total ~1.5M params. Single output channel (residual phase).

## Adımlar

### A. Veri pipeline (1-2 gün)
- [ ] `scripts/build_track_c_dataset.py`: GT manifest + outlier filter → 53 valid frame
  - Her frame için: ϕ_classical (poly5 reffree, fixed z) + ϕ_GT (ref-based, fixed z) + residual = ϕ_classical - ϕ_GT
  - `.npz` dump per frame (compressed) + master `dataset.json` index
  - Train/val/test split: leave-one-session-out (her session bir kere test'te)
- [ ] Augmentation: random flip H/V, random 90° rotate, random crop 768², piston offset
- [ ] Sanity check: bir frame'in residual'ini görselleştir, RMSE'sini logla, structured stripe hâlâ orada olmalı

### B. Model + training loop (2-3 gün)
- [ ] `src/recon_dl/unet_lite.py`: U-Net lite mimarisi (4-down 4-up, base=32, GroupNorm, GELU, skip connections)
- [ ] `src/recon_dl/dataset.py`: PyTorch Dataset wrapper, .npz lazy load + augmentation
- [ ] `src/recon_dl/losses.py`: L1 + SSIM (`pytorch-msssim`) + TV combine
- [ ] `scripts/train_track_c.py`: training loop, AdamW lr=3e-4 cosine decay, 200 epoch, batch=4 (M-series MPS) / 16 (CUDA), early stop on val RMSE
- [ ] Checkpoint: `models/track_c/v0.1/{model.pt, train_log.json, config.yaml}`

### C. Evaluation (1 gün)
- [ ] `scripts/eval_track_c.py`: held-out test session üzerinde RMSE/p95/SSIM/inference latency
- [ ] LOSO cross-validation: 9 session × 9 model (her session sırayla test), her run sonucunu `eval_loso.csv`'ye yaz
- [ ] Karşılaştırma raporu: `_benchmark_track_c/report.png` — Track A vs Track C, frame-level RMSE distribution + delta visualization
- [ ] Pass/fail: median RMSE ≤ 0.50 rad? p95 ≤ 1.50? Yoksa hangi sample tipinde başarısız?

### D. Production wiring (1-2 gün, B+C geçtiyse)
- [ ] `src/core/pipelines/reffree_hybrid.py`: classical pipeline wrap + CNN inference plug-in
- [ ] Model serving: TorchScript export + lazy load (CPU fallback)
- [ ] UI flag: ProcessTab "Reference-free reconstruction (CNN-corrected)" checkbox
- [ ] Telemetry: her CNN inference için input/output statistics audit log'a; drift için
- [ ] Smoke test: live UI'dan referanssız mod aktif → 5 frame koş → RMSE histogram göster

### E. Docs
- [ ] `docs/REFFREE_HYBRID.md`: mimari, training rehberi, retrain trigger'ları
- [ ] `tasks/lessons.md`: training sırasında karşılaşılan tuzaklar (her major correction sonrası)
- [ ] `CHANGELOG.md`: feature entry

## Zaman tahmini
- A: 1-2 gün
- B: 2-3 gün
- C: 1 gün
- D: 1-2 gün
- E: paralel
- **Toplam: 5-8 iş günü**

## Risk & mitigation

| Risk | Olasılık | Etki | Mitigation |
|---|---|---|---|
| 53 frame yetersiz, overfit | orta | yüksek | Augmentation agresif + dropout 0.1; eğer overfit, synthetic stripe injection augmentation ekle |
| Cross-session generalization zayıf | orta | yüksek | LOSO CV ile erken yakala; başarısızlıkta session-conditional film/adain ekle |
| Inference latency >100 ms | düşük | orta | Model boyutunu azalt (base=24 veya 16); ONNX/CoreML export |
| Sample-tipi farklılığında degrade | orta | orta | Production drift monitoring + Track A fallback; lab veri kampanyasıyla yeni sample örnekleri ekle |
| GT manifest 10 frame outlier var | bilindik | düşük | `--filter-z-outliers` ile zaten ayıklandı (53/63 valid) |

## Definition of done
1. ✅ LOSO median RMSE ≤ 0.50 rad
2. ✅ p95 abs err ≤ 1.50 rad
3. ✅ Inference < 100 ms (CPU, 1024²)
4. ✅ Production wired + UI flag
5. ✅ docs + lessons güncel
6. ✅ Track A fallback hâlâ çalışıyor (CNN bypass mod)

## Out of scope (Track C kapsamı dışı, daha sonra)
- Multi-sample augmentation (RBC + bakteri synthetic injection) — yeni veri geldiğinde
- Track B pivot — 5000+ frame 8+ sample tipi geldiğinde
- Real-time GPU inference optimizasyonu — production'da CPU yetiyorsa atla
- Tek shot autofocus + reffree birleşik DL (autofocus de CNN'e gömülsün) — bir sonraki sprint

---

# Backlog — 2026-07-05 (kullanıcı notları)

## 1. "Adaptive" konseptli autofocus algoritmalarını oturt
Kullanıcı: "adaptive konseptli algoritmalarımız vardı, onları oturtamadık tam — bir ara oturtalım."
- Kapsam: `src/core/autofocus/search_adaptive.py` — adaptive_gradient / adaptive_ratio / adaptive_bracketing / adaptive_distance + `AdaptiveFocusState` (canlı kamera state machine). Kök `adaptive_steps/` staging klasörü (Mar 2026 prototipi) hâlâ merge artığı olarak duruyor.
- Bilinen bağlam: B-023 (v1→v2 portunda adaptive algoritmalar kaybolmuştu, geri eklendi); v2 `workers.py` 6 algoritmayı listeliyor ama adaptive'lerin gerçek lab verisinde güvenilirliği hiç sistematik ölçülmedi (bench_autofocus.py sentetik tek-küre ile koşuyor).
- Yapılacak (önerilen): (a) gerçek lab hologramlarında (labtest/ + rapor data) adaptive vs klasik benchmark; (b) hangi adaptive modun hangi manzara tipinde kazandığını belgele; (c) default'ları/parametreleri buna göre sabitle; (d) adaptive_steps/ staging klasörünü temizle.

## 2. Kitap-algoritma tutarlılık doğrulaması (başlatıldı 2026-07-05)
Optik/rekonstrüksiyon/mikroskopi/holografi kitaplarından analitik test vakalarını çıkarıp motorda tutarlılık testi — `tests/validation_textbook/` süiti.

## Review — 2026-07-05 oturumu (kitap-doğrulama + code review + fix sprint)

1. **Kitap-algoritma tutarlılık süiti** `tests/test_textbook_validation.py`: 13 PASS + 1 scope-skip.
   Kaynaklar Kim/Kreis/Hecht (gerçek PDF'ler); Born&Wolf ve Schnars&Jüptner dosyaları SAHTE çıktı
   (biri 3-sayfalık kitap incelemesi, biri indirme-scam placeholder) → formüller gerçek kaynaklardan
   çapraz-alındı. Yük-taşıyan formüller physics_verify (Docker oracle) yeşil. Rapor:
   tasks/textbook-validation-2026-07-05.md. Kapsam sınırı belgelendi (Fraunhofer far-field +
   single-FFT Fresnel pixel motor kapsamı dışı).
2. **Code review (8 açı + adversarial doğrulama)**: 10 bulgu → 7 CONFIRMED + 1 jog-guard düzeltildi,
   regresyon testli; B-055..B-063 registry'de. Rapor: tasks/code-review-2026-07-05.md.
3. **BONUS motor bug'ı (B-058)**: propagate() spektrum cache id-reuse zehirlenmesi — textbook süiti
   yakaladı, weakref-kimlikle düzeltildi. Batch/timelapse döngülerinde yanlış-frame rekonstrüksiyonu
   riskiydi.
4. Süit: **1171 PASS + 10 skip**; bug sweep **63 entry FAIL=0**. crash_handler'daki 3 hata ön-var-olan
   ortam sorunu (stash'li ağaçta da aynı) — ayrıca incelenmeli.
5. Açık: review #8/#9 (bilinçli dokunulmadı), reffree'nin core/pipelines'a çıkarılması (yapısal tema),
   qpi.py:418 OPD→radyan bug'ı (wiki'de işaretli, ayrı fix bekliyor — bu diff'in dışında).

## Devam (2026-07-05, ikinci tur) — qpi fix + reffree refactor

6. **qpi.py boyut bug'ı DÜZELTİLDİ (B-060)**: compute_cell_morphology'ye wavelength_m eklendi,
   φ=2π·OPD/λ (physics_verify'lı bağıntı); compute_qpi gerçek λ'yı geçiriyor. Test:
   test_qpi.py::test_cell_morphology_phase_stats_have_correct_scale. Wiki callout'ları güncellendi.
7. **Reffree layering fix (B-064)**: src/core/pipelines/reffree_hybrid.py oluşturuldu (OpticalConfig
   parametreli). src/recon_dl/inference.py artık core'dan import ediyor (scripts→src inversion kırıldı,
   kaynak-guard testli). benchmark_reffree + run_rapor_data_batch core'a delege eden ince wrapper
   (byte-parity testli). Testler: test_reffree_pipeline.py (7). Suite: 1179 PASS; sweep 65 entry FAIL=0.
   Kalan: batch_renderer/depth_map ref-division kopyaları, reffree testlerini tmp_path'e taşıma,
   ProcessTab reffree entegrasyonu.

## Devam (2026-07-05, üçüncü tur) — 4 subagent cleanup sprint'i (orkestrasyon + review)

4 paralel Sonnet-5 subagent + 2 reviewer subagent; orkestrasyon/onay/final doğrulama ana oturumda.
- **A (ref-division dedup):** `safe_reference_divide` → core/reconstruction.py; reffree_hybrid re-export;
  depth_map + batch_renderer._apply_ref (2 dal) helper'a bağlandı. Reviewer side-note'uyla ana oturum
  3 kalan kopyayı da bağladı (qpi.subtract_reference_wave, autofocus/evaluator ×2 [tekli+batched],
  reconstruction_worker._subtract_reference) → idiom artık TEK kaynak.
- **B (track_c test taşınabilirliği):** sentetik tmp_path fixture (gerçek şemayla birebir); bayat
  Windsurf model yolu → REPO_ROOT-relative (inference testi torch'lu env'de artık GERÇEKTEN koşuyor —
  sistem python3 + torch 2.11/MPS ile 7/7 doğrulandı, losses device fix'i MPS'te kanıtlı);
  yeni torch'suz tests/test_track_c_dataset_schema.py; DHM_TRACK_C_DATA env override.
- **C (scalebar birleştirme + bg basis cache):** main_window._nice_scalebar_length silindi →
  core.scalebar.compute_scalebar (image-panel'le tutarlı; not: sub-µm etiketler artık '0.3 µm',
  '300 nm' değil); background_phase Zernike/poly basis+grid lru_cache(4) — sayısal birebir,
  3000-frame reffree batch'te frame-başına ~230-640MB yeniden-kurulum yok.
- **D (device_panel):** çift kurulum yolu kaldırıldı (__init__ gerçek çözümleme, __new__ hilesi silindi);
  per-tick threading.Timer → tek daemon thread + Event.wait + dirty-check publish.
- **Review turu (2 reviewer):** core-fizik değişiklikleri TEMİZ (0 bulgu; divide bit-for-bit, cache
  mutasyonsuz, circular-import her sırada yok). UI reviewer 1 HIGH yakaladı: stop_polling'in _closed
  latch'i reopen'da HUD'u kalıcı donduruyordu → ana oturum düzeltti (start her seferinde taze Event,
  loop kendi event referansını taşır — clear() yarışı imkânsız) + reopen regresyon testi.
- Süit: **1191 PASS + 10 skip**; bug sweep 65 FAIL=0. Düşük bulgular kabul edildi: px_um≤0 latent
  guard (spinbox floor'u nedeniyle erişilmez, yeni davranış daha güvenli), fixture helper'ın iki test
  dosyasında bilinçli kopyası (docstring'lerde senkron notu).

## Phase 3 (2026-07-05) — ui2 reference-mode UI + reffree pipeline yolu

Plan: docs/AI_VISION_MCP_PLAN.md ("Phase 3 — ui2 reference-mode UI").

- [x] `ReconParams` (src/ui2/reconstruction.py): yeni alanlar `reference_mode`
  ("off"|"reference"|"reference_free"), `reffree_bg_method`, `reffree_bg_order`,
  `reffree_n_terms`, `reffree_cnn`. `effective_reference_mode()` helper: mode=="off"
  ve legacy `subtract_reference=True` ise "reference" döner (geriye uyum, tek yer).
  `_extract_field_with_reference` + `_prepare_sample_and_ref_fields` bu helper'a
  bağlandı (grep edilen tüm subtract_reference tüketicileri).
- [x] `ReconResult`e `unwrapped_phase` (Optional) + `reffree_note` (Optional[str])
  eklendi — sadece reference_free modunda dolduruluyor, "off"/"reference" hiç
  dokunmuyor (wrapped `phase` degismedi, geriye uyum tam).
- [x] `ReconstructionDriver._run`: reference_free ise unwrap (params.unwrap_method) +
  `core.background_phase.subtract_background` (reffree_bg_method/order/n_terms) →
  `unwrapped_phase`. CNN toggle: `reffree_cnn_available()` (torch importable VE
  models/track_c/v0.1/model.pt var) guard'lı, lazy import, hata/unavailable →
  `reffree_note` ile status'a yansir (asla sessiz cokme/no-op).
- [x] Depth-map yolu: `_prepare_sample_and_ref_fields` reference_free'de
  `ref_field=None` döner (bg fit derinlik taramasına uygulanmıyor — bilinçli
  kapsam disi, docstring'de belgeli).
- [x] app.py UI: "Reference mode" combo (Off/Reference/Reference-free) +
  reference_free icin bg-method combo + order input + CNN checkbox (gate'li,
  tooltip "requires torch + trained model"). `_on_param_changed`,
  `_hydrate_widgets`, `_snapshot_state`, `_apply_preset`, `_compose_info_text`
  (yeni "Ref mode:" satırı, eski "Reference: … (on/loaded/none)" satırı
  DOKUNULMADI — mevcut testler kırılmadı) güncellendi.
- [x] Persistence: `Ui2State` + `SCHEMA_VERSION` 12→13 no-op stamp migration
  (`state_store._v12_to_v13`), settings_schema.py yorum + alanlar.
- [x] Testler: tests/test_ui2_reffree.py (24 test) — ReconParams defaults/back-compat,
  gercek sentetik off-axis hologram uzerinden reference_free kosumu (bg fit
  degistigini kanitlar), reference_free bir reference yuklu olsa bile ref
  division'i atlar, CNN gate (torch yok → zarif skip + note), state round-trip +
  migration, DhmApp combo label/handler.

Kosuldu: `pytest tests/test_ui2_reffree.py tests/test_ui2_logic.py
tests/test_ui2_scientific_params.py tests/test_ui2_state_store.py
tests/test_settings_schema.py tests/test_batch_v2_parity.py` → **106 PASS**.
Tam suite: 1249 PASS + 10 skip, 3 FAIL (test_crash_handler.py — degisiklik
oncesi de ayni sekilde fail ediyor, bu gorevin disinda, dokunulmadi).

## AI Vision + MCP + Reffree UI — 3 faz (2026-07-05, multiagent) TAMAM

Plan: docs/AI_VISION_MCP_PLAN.md. Karar: core-first · MCP full-drive headless · vision numeric+PNG · reffree UI→ui2.
Workflow: Faz A (1b tools + 3 ui2 paralel) → Faz B (MCP) → 2 reviewer. Orkestrasyon + review-fix ana oturumda.

- **Phase 1 core/observe.py** (ana oturum): inspect_reconstruction/inspect_phase_map/inspect_field/render_view (saf, Qt-free). 15 test; gerçek PNG render doğrulandı.
- **Phase 1b tool wiring** (agent): 5 AI tool (inspect_*/render_view/set_reconstruction_mode) tool_impls'e; ToolContext.get_last_field/set_reference_mode (dondurulmus kontrat); v1 ai_panel wiring; render_view PNG'yi ~/.dhm-reconstruction/renders'a yazar, sonucta base64 YOK. 24 canonical tool (docstring guncel).
- **Phase 2 dhm-mcp** (agent): src/dhm_mcp/ headless MCP sunucu, build_tool_registry generic kopru + dispatch (clamp/schema/confirm/audit devrede), render_view→MCP image content, mcp'siz durust SystemExit.
- **Phase 3 ui2 reffree** (agent): ReconParams reference_mode(off/reference/reference_free)+reffree_bg/cnn; workers reference_free→background_phase.subtract_background; app.py 3-yonlu combo+bg/CNN gate; SCHEMA_VERSION 12→13.

**Review (2 gozden gecirici) + ana-oturum fix'leri:**
- **B-066 CRITICAL** (MCP): FastMCP tool schema'yi imzadan cikariyor → **kwargs handler'lar tum tool'lari cagirilamaz yapmisti; her ToolSpec JSON-schema'sindan explicit keyword-only imza sentezlendi. mcp'li ortamda 3 test canli dogrulandi (mcp-gated → manual).
- **B-067 HIGH** (ui2): reference-mode combo ↔ legacy subtract_reference iki-yonlu senkronsuzlugu (explicit Off flag'i birakmiyordu → divide devam) duzeltildi (combo/toggle/load/clear); + QPI yolu artik reffree bg-fit uyguluyor (MED); audit RESOLVED mode logluyor (LOW); hydrate honourable-olmayan CNN istegini dusuruyor (LOW).
- render_view dosya adina uuid suffix (mikrosaniye cakismasi).

Suite: **1261 PASS + 11 skip**; sweep 67 entry FAIL=0. core/observe thread-safety + unit-conversion + no-base64-leak reviewer'ca CLEAN.
Kalan (dusuk): render retention/cleanup yok; irreversible tool henuz yok (confirm-gate sentetik test'te); ui2 AI paneli hala bozuk (ayri is).

## ui3 — Qt yeniden inşası (2026-07-05, multiagent)

Kullanıcı: "ui2'yi sıfırdan inşa et, arayüz paketini sen seç, hiçbir parçayı atlama."
Toolkit KARARI (kolayına gelen değil, değerlendirilmiş): **PySide6 (Qt6) + pyqtgraph**.
Web/Tauri reddedildi (numpy compute'a IPC vergisi); Dear PyGui reddedildi (bug registry'nin
en büyük fazı DPG_PORT 17 macOS bug'ıydı). Gerekçe: docs/UI3_DESIGN.md.

- **Yeni paket src/ui3/** (31 dosya, 8145 satır): tasarım sistemi (design.py token'lar + qss,
  4 palet WCAG-AA), wcag.py, state.py (persist+migration, ReconParams reuse), bridge.py
  (Qt-free ui2 ScienceDriver/ReconstructionDriver üzerine QThread/Signal köprüsü — compute
  YENİDEN YAZILMADI), viewport.py (pyqtgraph ImagePanel zoom/pan/colormap/scalebar/drop),
  main_window.py (dock shell, 4-panel grid, tam menü, workflow modları, ⌘K palette, tema,
  maximize, onboarding), context.py (dondurulmuş PanelContext kontratı).
- **11 panel/dialog multiagent ile paralel inşa** (kontrata karşı): recon (zengin, tüm ReconParams
  + reference-mode off/reference/reference-free + CNN gate + presets), focus, qpi, depth,
  camera, device, report, timelapse, ai (Qt-native AIWorker + vision render_view inline),
  + dialoglar (surface 3D pyqtgraph.opengl, qpi_batch, focus_candidates, audit_viewer,
  onboarding, preset_dialogs, line_profile, preset_chips).
- **Entegrasyon (elle):** inline dock → ReconPanel (source-of-truth self._params), 8 feature
  panel dock (Analyse'da tabbed), dialoglar menü/sonuç-tetikli, depth.surface_requested →
  SurfaceViewer, onboarding first-run.
- **Kapsam matrisi (docs/UI3_DESIGN.md) 1:1 karşılandı** — hiçbir ui2 parçası atlanmadı.
- run_ui3.py giriş. Eski ui2 (Dear PyGui) parite gelene kadar DOKUNULMADI.
- Test: **131 ui3 testi** (offscreen Qt, gerçek QApplication — DPG-stub yok); tam repo **1392 PASS**.
- Kalan: gerçek Mac ekranında görsel doğrulama (offscreen GL context surface-viewer'ı çizmiyor,
  fallback var); ui2 emekliye ayırma (parite teyidinden sonra); adaptive-autofocus backlog hâlâ açık.

---

## Review — 2026-07-06: ui3/observe/dhm_mcp 2. kod-review turu (B-072…B-082)

Kullanıcı: "code review + bulguları düzelt." 8-açılı finder + adversarial doğrulama turu
ui3+observe+dhm_mcp'yi ilk kez bu derinlikte taradı → **11 gerçek bug** (birkaçı ui3
entegrasyonundan). Hepsi düzeltildi, regresyon yazıldı, bug registry'ye işlendi.

- **B-072 (ÇÖKME)** `main_window._on_load_reference` silinmiş `_cb_ref_mode` combo'suna
  `setCurrentText` → AttributeError. Fix: params + `_sync_controls_from_params()`.
- **B-073 (KRİTİK)** `panels/ai_panel` bridged tool'ları argümansız `signal.disconnect()`
  → sinyaldeki tüm slot'ları koparıyor (viewport paint + paneller). Fix: `_disconnect_one(handle)`.
- **B-074** `core/observe.render_view` scalebar downsample stride'ı yok saydı → etiket
  stride kadar yanlış. Fix: `_downsample` stride döndürür, `pixel_size_um*stride`.
- **B-075** `ui3/state._params_to_dict` `str(tuple)` → `af_roi` round-trip bozuk. Fix:
  JSON list + yüklemede tuple coerce.
- **B-076** `ui3/bridge.busy_changed` iki executor arası tek boolean → erken idle. Fix:
  lock'lu referans-sayımı.
- **B-077 (B-067 sınıfı)** AI `_gui_set_reference_mode` legacy `subtract_reference`'ı
  senkron tutmuyordu → bayat True açık "off"u eziyor. Fix: `= mode=="reference"`.
- **B-078** headless `set_recon_param` türetilmiş cache invalidate etmiyor → bayat inspect.
  Fix: cache temizle.
- **B-079** MCP'de referans modu erişilemezdi (`reference_raw` dolmuyor). Fix: opsiyonel
  `reference_path` (validate_path'li) → headless + ui3 AIPanel yükler.
- **B-080** reference-free tek `bg_order` iki knob'u besliyordu → polynomial'de dejenere.
  Fix: method-özel default + doğru knob.
- **B-081** AIPanel health thread öksüz bırakma → "QThread destroyed while running". Fix:
  probe set izleme + `shutdown()` hepsini bekler.
- **B-082 (çift-wiring)** shell + paneller aynı sinyale bağlı → autofocus çift-compute,
  depth çift-repaint. Fix: shell auto-reconstruct'ı bıraktı, depth'i yalnız cache'ler.

**Doğrulama:** `tests/test_ui3_review_2026_07_06.py` (11 yeni regresyon, tümü PASS) +
ilgili süitler. Tam repo **1400 PASS**, 11 skip. 3 `test_crash_handler` başarısızlığı
**pre-existing** — bayat `venv/pyvenv.cfg` "Windsurf Projects" yolundan test yükleme
harness artefaktı, dokunulan koddan bağımsız (venv rebuild ayrı iş). Obsidian
[[DHM-ui3-Qt-Rebuild]] güncellendi.

---

## Devam — 2026-07-06 (2. seans): B-083/084/085 + venv rebuild → TAM YEŞİL

- [x] **B-083** — 17 bulgudan kalan son #15: QPI/candidates çift-wiring. Sahiplik kontratı:
  paneller domain statüsü (one-shot QPI + dry mass), dialoglar batch/candidates statü+tablo,
  shell yalnız recon-paint + z-sync + cache + `present()` (repopulate'siz göster); shell'in
  error lambda'ları kaldırıldı. +2 regresyon testi.
- [x] **Venv rebuild** — `pyvenv.cfg` kanıtı: venv `Windsurf .../Phyton/venv` olarak yaratılıp
  kopyalanmış (taşınamaz). pip freeze yedeği → aynı yorumlayıcı (3.13.5) → aynı pinli 56 paket
  → tüm `__pycache__`/`.pytest_cache` temizliği. Eski venv silindi.
- [x] **B-084** — 3 crash_handler hatasının GERÇEK kökü (bayat-pyc değil): pytest-qt (ui3 ile
  geldi) handler'ın doğru excepthook zincirlemesini "Exceptions caught in Qt event loop" diye
  yalancı FAIL'e çeviriyordu. Testler zararsız önceki-hook sabitliyor → 7/7.
- [x] **B-085** — `from fixtures...` koleksiyon-sırası şansı: hiç toplanamayan
  test_autofocus_speed_baseline dahil 6 dosya kendine-yeter yapıldı; `--ignore` kalktı.
- [x] Registry: **85 kayıt, FAIL=0** (`check_bugs.py`). Tam süit (ignore'suz): **1411 PASS,
  0 FAIL, 11 skip** (torch/mcp/ffmpeg opsiyonelleri).
- [~] Bugünkü tüm diff'in adversarial review workflow'u (4 boyut × 2 refuter) — sonuç bekleniyor.

## Review turu #2 — 2026-07-06: kendi fix diff'ime adversarial workflow (B-086…B-094)

Bugünkü B-072..B-083 diff'i 4-boyutlu finder + 2'şer refuter'lık workflow'la (32 agent)
bağımsız incelendi → 8 onaylı + 3 belirsiz bulgu; belirsizlerin ÜÇÜ de elle doğrulanıp gerçek
çıktı. Hepsi düzeltildi (9 benzersiz fix, +9 regresyon testi):
- [x] B-086 headless: set_reference_mode + invoke_autofocus'un z yazımı cache invalidation'ı
      atlıyordu (B-078'in eksik yüzü) → tek `_invalidate_derived()` her mutator'da.
- [x] B-087 ui3 AI snapshot HAM kamera pikselini geçiyordu → dry mass M² (40x'te 1600×) şişik,
      scalebar M× yanlış → snapshot artık effective_pixel_um().
- [x] B-088 AI tool'ları senkron busy-reddini kaçırıp 60-120s timeout'a asılıyordu → `if not got` guard'ı.
- [x] B-089 B-076 refcount'un kendi yarışı: worker'dan kuyruklanan idle, yeni busy'den sonra
      inip UI'ı temizliyordu → idle kararı GUI thread'de teslim anında (_idle_check sinyali).
- [x] B-090 set_status auto-toast + panellerin explicit toast'u = çift toast → status-only.
- [x] B-091 ui3 AI bg_order zernike'de sessiz no-op (yanlış knob) → method'a göre yönlendirme.
- [x] B-092 is_cancelled closure worker'ı atamadan ÖNCE yakalıyordu → Stop çalışmıyordu → dinamik okuma.
- [x] B-093 observe spectrum karmaşık girişte fft2 atlıyordu (|field| log|F| diye çiziliyordu) → tek fft2 yolu.
- [x] B-094 get_field('phase_unwrapped') off/reference modda SARILI fazı sessizce veriyordu
      (yalnız reffree unwrap dolduruyor) → talep-üzerine unwrap + cache.

---

## Adaptive Autofocus — OTURDU (2026-07-06, B-095)

2026-07-05 backlog maddesi (satır 1531-1534) kapandı; plan (a)-(d) tamamı uygulandı:
- [x] (a) **Gerçek-veri benchmark**: 9 lab sahnesi (session_01..09 orta kareleri, 1600×1200,
      lab optiği) × 2 metrik × 6 algoritma; gerçek = 201-adım yoğun tarama ±20mm; bütçe = 40 eval.
      Harness `scripts/benchmark_af_real.py`, ham veri `tasks/af_real_benchmark.json` (108 satır).
- [x] (b) **Sahne/metrik verdiktleri belgelendi** (`docs/AUTOFOCUS_ADAPTIVE.md` + UI tip'leri):
      robust iki metrikte de tepe (%78/%67 hit, 0.07/0.15mm); adaptive_bracketing laplacian
      doğruluk şampiyonu (0.03mm/%78, bütçe +%30); adaptive_gradient hızlı ama ENTROPY ile
      GÜVENİLMEZ (7.2mm — düz-omuz stall'ı); adaptive_distance bu kurulumda güvenilmez (%0-22);
      ESKİ default zscan en kötü ikinci (%33/%44 — grid aralığı = doğruluk tabanı).
- [x] (c) **Default'lar sabitlendi**: ReconParams + settings şeması af_algorithm="robust"
      (v9 migration zscan'de DONMUŞ — mevcut state dosyaları davranış değiştirmez); headless/MCP
      autofocus adaptive_gradient→robust + default metric PHASE_VARIANCE→LAPLACIAN_VARIANCE
      (GUI paritesi); yeni ReconParams knob'u YOK (darboğaz algoritma+metrik eşleşmesi, iç
      parametreler değil — ui3 FocusPanel docstring'i "settled" olarak güncellendi).
- [x] (d) **`adaptive_steps/` staging klasörü silindi** (Mart 2026 prototipi, git rm ile staged).
- [x] Regresyon: `tests/test_af_settlement.py` (5 test). Registry: **B-095**.

---

## Driver Relocation — 2026-07-06: ui2 → core/drivers (ui2 emekliliğinin ön koşulu)

- [x] `src/ui2/{reconstruction,workers,camera_feed}.py` → `src/core/drivers/` (git mv,
      geçmiş korunarak; modül adları aynı → içteki göreli importlar sıfır değişiklik).
- [x] Eski yollar **sys.modules-aliasing shim**: `ui2.workers` ile `core.drivers.workers`
      kelimenin tam anlamıyla AYNI modül nesnesi → `patch("ui2.workers.X")` gerçek
      global'leri vurmaya devam eder, private isimler + sınıf kimliği korunur.
- [x] Dış tüketiciler core.drivers'a çevrildi: ui3 (context/main_window/bridge/state +
      camera/recon/focus panelleri) + `core/cameras/synthetic.py` (core→ui2 ters katmanlama
      da düzeldi). **ui3 ve core artık ui2'yi HİÇ import etmiyor** (test pin'li).
- [x] Regresyon: `tests/test_driver_relocation.py` (alias kimliği, patch-through,
      katmanlama pin'i). Tam süit: **1429 PASS, 0 FAIL, 11 skip**.
- [~] Relocation diff'ine adversarial review workflow — sonuç bekleniyor.
- Kalan (ui2 emekliliği için): kapsam-parite teyidi + kullanıcı onayı; ui2 dizini artık
  yalnız DPG sunum katmanı + shim'ler.

### Relocation review sonucu (2026-07-06): 2 onaylı bug, ikisi de düzeltildi
- [x] **B-096 (HIGH):** workers.py bir seviye derine taşınınca `_REPO_ROOT parents[2]` →
      `<repo>/src`'a çözüldü; Track C CNN checkpoint yolu bayat → `reffree_cnn_available()`
      sessizce False (geçerli checkpoint varken CNN gri). Fix: `parents[3]` + yol pin testi.
      DERS: dosya taşımasında __file__-göreli yolları MUTLAKA yeniden denetle.
- [x] **B-097:** yeni `core/drivers/__init__` eager `.workers` importu → `import
      core.drivers.camera_feed` (numpy-only modül) matplotlib+skimage+tüm bilim yığınını
      çekip ~0.7s'e çıktı; ui2 shim'leri de regresyonu miras aldı. Fix: PEP 562 lazy
      __getattr__ (ui2/__init__ kalıbı) + hafiflik subprocess testi.
- [x] Relocation test dosyası 7 teste çıktı; registry B-096/B-097.

---

## ui2 EMEKLİ — 2026-07-06 (kullanıcı onayı: "kaldır")

- [x] **DPG sunum katmanı silindi** (13 modül: app/theme/image_panel/surface/widgets/dialogs/
      device_panel/ai_panel/ai_bridge/ai_panel_state/line_profile_state/ui_state/wcag) +
      `run_ui2.py`. Git geçmişinden geri getirilebilir. dearpygui zaten ne requirements'ta
      ne venv'deydi (testler stub'luyordu — ui2 bu ortamda çalışamıyordu bile).
- [x] **Kalanlar**: `ui2/__init__` (emeklilik notu), `ui2/state_store.py` (kalıcı ayarlar +
      DONMUŞ v1..v10 migration'ları — disk-uyumluluğunun tek kaynağı), 3 driver shim'i.
- [x] **Test triage**: 7 salt-DPG dosyası + test_v21y_ui_polish silindi; v209'un WCAG bölümü
      ve test_scalebar'ın DPG testi kesildi; 5 karışık dosya paralel agent'larla cerrahî
      ayıklandı — **51 driver/state testi korundu** (reffree gating, preset persistence,
      migration'lar, ROI, QPI halving), 48 DhmApp testi kaldırıldı. Preset kapsamı ui3
      recon_panel testlerinde yaşıyor (doğrulandı).
- [x] **Registry**: silinen DPG koduna pinli 7 tarihsel kayıt (B-011/026/028/031/057/060/067)
      not düşülerek manual'a çevrildi — düzeltilen kod silindiği için gerileyemezler.
- [x] CHANGELOG'a emeklilik bölümü eklendi.
- [x] Tam süit: **1194 PASS, 0 FAIL, 11 skip** (238 DPG testi emekliyle gitti).
      Registry: 97 kayıt, FAIL=0.

---

## 3-Versiyon Denetimi — 2026-07-08 (kullanıcının en baştaki hedefi kapandı)

Kullanıcı: "bu programın başka versionları da var... daha sonra diğerlerini de inceleriz."
Julia + Phyton, Hybrid'e karşı 6-ajanlı workflow ile denetlendi (5 okuyucu + skeptik sentez).

**Soyağacı:** Phyton = orijinal PySide6 app (Hybrid'in atası) AMA çekirdeği juliacall RPC shim'i
(gerçek fizik Julia CoreModule'de); Julia = Phyton'un yarım-kalmış deneysel portu (~360 satır);
Hybrid = keeper, neredeyse temiz üst-küme. Kararlar: Hybrid keep-active / Julia archive-dead /
Phyton salvage-then-archive. Detay: Obsidian [[DHM-Version-Audit-Phyton-Julia-Hybrid]].

**Salvage: 10 ham aday → 3 elemeyi geçti:**
- [x] **B-098 (do-now):** Phyton'un düz-eğri/non-finite autofocus teşhisi Hybrid'in çıplak
      `autofocus_zscan`'ine taşındı. Mis-parametrede sessiz argmax-of-noise yerine artık
      `focus_landscape_warning()` → `AutoFocusResult.warning` (raise değil, en-iyi tahmin korunur)
      → worker→FocusPanel "warn" statü + ⚠. 8 test. Flatness mantığı Hybrid'de zaten vardı ama
      yalnız adaptive_distance_search'te.
- [ ] **backlog (task-chip):** spectrum-tıkla +1-order merkez override (ui3 hook; çekirdek hazır).
- [ ] **backlog (task-chip):** gerçek USAF-1951 seti (data/220825, zaten repoda, kullanılmıyor)
      → regresyon testi. Hybrid şimdiye dek yalnız sentetik round-trip doğruladı.
- [x] skip (6): Julia kernel tweak'leri, MAD spike, endpoint-nudge, export-crop — hepsi ya
      zaten karşılanıyor ya marjinal.

**KORU:** `Hybrid/data/220825` — üç versiyondaki tek gerçek çekilmiş korpus + fizik ground-truth.

Süit: **1202 PASS, 0 FAIL, 11 skip**. Registry: 98 kayıt, FAIL=0.

---

## Version-audit backlog kapandı — 2026-07-08 (iki task-chip)

- [x] **B-099 — Spectrum-tıkla +1-order merkez override (ui3).** Çekirdek `center_yx`'i kabul
      ediyordu ama 3 OffAxisParams çağrı yerinin HİÇBİRİ ReconParams'tan geçirmiyordu + alan
      yoktu. Fix: `ReconParams.offaxis_center` + tek `_offaxis_params()` helper (reconstruct/
      qpi/autofocus+depth); ui3 spectrum-viewport tıklama (Process ▸ Pick +1 order, tek-atım,
      crosshair marker+cursor) → param + yeniden-reconstruct + Reset-to-auto; tuple kalıcı. 10 test.
- [x] **Gerçek USAF-1951 regresyon testi** (`tests/test_real_usaf_reconstruction.py`, `slow`,
      veri yoksa skip). Config türetildi: kamera pikseli 3.45µm @ z=43mm ASM → orijinal-app
      referansına korr **0.71** (300mm'de 0.56, odak-duyarlı); yalnız genlik (USAF genlik hedefi,
      faz korr 0.04 → assert edilmedi); + gerçek-veri ASM round-trip 9.8e-7 + off-axis +1-tespit
      sağlaması. `data/220825` (75MB) artık load-bearing korpus.

---

## Version-audit salvage — 2026-07-08 devam (B-100 + kalanların değerlendirmesi)

- [x] **B-100 — default 'robust' path'inde flat/non-finite uyarısı ölüydü.** Audit'in
      "HIGH confidence" salvage'ı (flat-curve tanısı) aslında ZATEN vardı (B-098,
      `focus_landscape_warning`) — ama driver `getattr(core_result,'warning',None)` yaptığı
      için yalnız linear zscan taşıyordu; settled default 'robust' (B-095) + coarse_to_fine/
      golden/adaptive hep None dönüyordu → yanlış-parametreli run default path'te hâlâ SESSİZCE
      dejenere oluyordu. Kök-neden fix: `_landscape_warning()` helper her aramanın tuttuğu
      landscape'ten hesaplıyor (robust → uniform coarse_z/coarse_scores; golden/coarse_to_fine
      landscape tutmaz, adaptive trace non-uniform → false-positive'i önlemek için bilinçli None).
      5 yeni test (uçtan-uca robust dahil). Körü körüne port etseydim mevcut mantığı çoğaltır,
      asıl boşluğu kaçırırdım — doğrulama kazandırdı.

### Kalan salvage adayları — DEĞERLENDİRİLDİ, kullanıcı kararına bırakıldı (silent-fix YOK)
- **Evanescent decay (ASM kernel, physics, medium-conf):** Julia `exp(-|Im(kz)|·|z|)` uyguluyor;
  Hybrid radicand'ı 0'a clamp'leyip evanescent bandı H≈1 geçiriyor. AMA asıl NaN-büyüme bug'ı
  Hybrid'de zaten çözülü (|H|≤1 clamp, reconstruction.py:99-105). Bu bir *iyileştirme* (yüksek-NA/
  büyük-z'de gürültü sızıntısı), bug değil. ASM çekirdeği 1200+ testin bağlı olduğu taç mücevher →
  medium-conf bir refinement için dokunmak riskli. **Öneri: ayrı, dikkatli bir oturumda + physics_verify ile.**
- **Endpoint-avoidance nudge (trivial, medium):** audit'in kendisi `adaptive_distance_search`'ün
  "asıl çözüm" olduğunu not ediyor → linear path'e marjinal katkı. Düşük öncelik.
- **MAD impulse rejection (small, low):** audit "Hybrid'in phase-domain metrikleri + reference
  division zaten z~0 spike'ına bağışık mı, doğrula" diyor → önce doğrulama gerek, düşük güven.
- **mod(phase,2π) before exp (trivial, low):** complex64 precision mikro-iyileştirme; ölçülebilir
  etki kanıtı yok. Düşük öncelik.

---

## CLAUDE.md-lens review — 2026-07-08 ("uygulamayı yeni claude.md'ye göre elden geçir")

7 alt-sistem finder × CLAUDE.md lensi (silent-degrade / kök-neden / uydurma-API / staff-bar)
→ her bulguya 2 çürütücü → **sentez** aşaması (find→verify→SENTEZ; kullanıcı düzeltmesi).
10 onaylı + 2 belirsiz → 4 sistemik desen. Baskın risk: normal-görünüp niceliksel YANLIŞ üreten
silent-degrade ailesi (özellikle reference-düzeltmesinin 3 bağımsız yoldan sessizce atlanması).

### fix_now — DÜZELTİLDİ (B-101..B-107, 8 test, süit yeşil)
- [x] **B-101 (SECURITY):** LLM sample_id path-traversal → charset guard (state-dir kaçışı kapandı).
- [x] **B-102:** ReconPanel "referans yüklenmedi" uyarısı info satırıyla eziliyordu → else branch.
- [x] **B-103:** coarse_to_fine golden fine-fazına roi_bounds geçmiyordu (ROI autofocus tam-kare
      optimize ediyordu) → forward edildi.
- [x] **B-104:** state reference_path (Path) round-trip'te düşüp mode="reference" kalıyordu
      (sessiz referanssız QPI) → Path str olarak kalıcı + load'da Path'e coerce.
- [x] **B-105:** _extract_evaluations robust'un total_evaluations'ını okumuyordu (default algo
      eval sayısını ~yarı raporluyordu) → total_evaluations de onurlandırılıyor.
- [x] **B-106:** depth cluster segmentasyon swallow'u log'suzdu → _LOG.exception eklendi.
- [x] **B-107:** headless recon summary phase_std_rad emit ediyordu; timelapse extractor phase_std
      okuyor (MCP timelapse faz-drift sinyalini sessizce kaybediyordu) → unsuffixed alias'lar.

### flag_for_decision — "devam" ile net olanlar DÜZELTİLDİ, gerçek-karar olanlar bekliyor
- [x] **B-108 (qpi.py:371):** compute_cell_morphology n_sample==n_medium'da hardcoded Δn=0.043
  (default kontrast) uyduruyordu → guard'lı kardeş `opd_to_height`'e bağlandı (Δn≈0→raise +
  tek-kaynak). compute_qpi'nin log'suz bare-except'i WARNING'e çevrildi (uncertain-2 kapandı). 2 test.
- [x] **B-109 (camera_feed.py):** AcquisitionThread hata kanalı yoktu → additive `on_error`
  callback + CameraPanel signal/slot (error status/toast + kontrolleri "stopped"a resetler). 2 test.
### "hepsini yapalım sırayla" — 3 gerçek-karar maddesi de DÜZELTİLDİ
- [x] **B-110 (workers reference-division runtime-surfacing):** aile'nin 3. yolu (config B-102,
  persistence B-104, bu = runtime). `_extract_field_with_reference` + `_prepare_sample_and_ref_fields`
  artık actionable not döndürüyor; ReconResult.reference_note (reconstruct), `_prepare_field` 4-tuple
  → AutofocusResult.warning (`_join_notes` ile birleşik) / QPI / QPIBatch / MultiFocus / DepthMapWrap
  .warning; shell recon handler + qpi/depth/focus panellerinde warn status+toast. Referans runtime'da
  patlarsa artık "başarılı" boyanmıyor. 6 test.
- [x] **B-111 (auto_select_metric):** peaksiz (monoton/kenar) metrik eğri-yüksekliğiyle ~1.0 puanlanıp
  gerçek-peak'li metriği yeniyordu → peaksiz=0.0 (güvenilirlik==prominence). Blast-radius küçük
  (yalnız v1-GUI opt-in "auto-select" toggle'ı; B-095 default path'i değil — doğrulandı). 1 test.
- [x] **B-112 (observe cell_count):** `segment_cell_phase` binary+keep-largest olduğu için count
  yapısal 0/1'di (çok-hücre → AI'ya "1 hücre"). Doğrulandı GERÇEK (uncertain haklıymış). Fix: observe
  kendi threshold+connected-component label'ı (speck-filtresiyle) → gerçek sayı; dry mass tüm hücreler
  üzerinden. 1 test (mevcut testi ==2'ye güçlendirdim).
- **Sonuç: `flag_for_decision`'ın 5 maddesi de kapandı** (B-108/109 + B-110/111/112). Sessiz-fix yok —
  hepsi verify-önce (B-111 blast-radius, B-112 "gerçekten binary mi").

---

## ui3 gerçek-ekran UX turu — 2026-07-10 (kullanıcı: "arayüzde çok hata var")

Kullanıcının gerçek oturumunu screenshot'layarak teşhis (offscreen'de görünmüyorlardı — bug'lar
RESTORE edilmiş state + native render + tab davranışındaydı). İmleçle interaktif doğrulandı.

- [x] **B-113** — 8 panel scroll-area'da değildi (içerik kırpılıyordu) + dialog'lar konumsuz açılıyordu
  (üst-üste). Fix: QScrollArea wrap + `present_centered` (parent'a ortalı + cascade).
- [x] **B-114** — hologram yükleme sonrası ReconPanel "(no hologram loaded)" kalıyordu → `_load_path`
  panel label'ını refresh ediyor.
- [x] **B-115** — `restoreState` bayat/uyumsuz dock-layout'unu geri yükleyip dock'ları merkezi grid'in
  üstüne yüzdürüyordu → `_LAYOUT_VERSION` damgası (uyuşmazsa layout atlanır, temiz düzen).
- [x] **B-116** — float edilen dock'u geri koymanın yolu yoktu ("çıkardım kaldı öyle") → "Reset panel
  layout" komutu (View menü + ⇧⌘0 + palette): un-float + eve dock.
- [x] **B-117** — 8 dock tek gruba tabify + mode'un sadece setVisible'ı → Qt tab-bar'ı bozuluyor
  (Analyse'da sadece Autofocus render, qpi/depth/ai ulaşılamaz "arkada"). Fix: mode-geçişinde görünen
  dock'ları tek temiz tab grubuna YENİDEN tabify + tab bar üste (North). Canlı doğrulandı: Analyse'da
  "Autofocus|QPI|Depth|AI copilot" tab bar'ı, tıkla-geç çalışıyor.
- İki şey kullanıcının kasıtlı eylemiydi (bug değil): sarı tema (high_contrast — geri alındı) ve dock'u
  incelemek için float etmesi (çıkış-yolu B-116 ile eklendi).
- **Ana ders (lessons.md):** offscreen test, RESTORE edilmiş kullanıcı state'ini ve native render'ı
  kanıtlamaz; state/tema/tab-bağımlı UI bug'ları için gerçek oturumu gör. Ve kullanıcı-ayarını "bug"
  sanıp izinsiz değiştirme.

---

## ui3 görsel polish — 2026-07-10 (kullanıcı: "testleri yap, arayüzü toparla iğrenç olmasın")

- [x] **Testler** — full suite 1253 passed / 11 skipped / 0 fail; ui3 spine 12/12; yeni B-118 testi geçiyor.
- [x] **B-118** — QSpinBox/QDoubleSpinBox/QComboBox okları boş açık-gri kare olarak render ediliyordu
  (parametre panellerini ucuz/bitmemiş gösteren görsel gürültü). Kök neden: QSS'te web CSS
  border-üçgen hilesi (width:0 + border renkleri) kullanılıyordu — Qt bunu üçgen olarak ÇİZMEZ, native
  bloğu çizer. İzole bir widget harness'inde ampirik doğrulandı: (a) border-hilesi kare çiziyor,
  (b) `image:url(data:...)` HİÇBİR ŞEY render etmiyor (Qt URL'yi dosya yolu sanıyor), (c) gerçek PNG
  dosyası net üçgen çiziyor. Fix: `design._ensure_arrow_icons()` paletin muted renginde up/down/chevron
  PNG'leri çiziyor (Retina için 2x logical boyutta), renk+şekil-versiyonuna göre
  `~/.dhm-reconstruction/icons` altında cache'liyor; `build_qss` bunları `image:url()` ile referanslıyor.
  Sadece QApplication varken çalışır; headless (test) width:0 oka düşer, asla boş kareye değil.
  Doğrulandı: offscreen (dark + high_contrast) + 2x DPR (Retina) → net üçgenler.
- Kullanıcının canlı app'ı ESKİ stylesheet'te (disk state'inin ilerisinde in-memory workflow_mode=Analyse)
  → oturumunu bozmamak için izinsiz restart etmedim; tema/param'lar persist ediyor, polish bir sonraki
  açılışta gelecek.
- **Kapsam-dışı, raporlandı (fix yok):** her panel adı 3 kez görünüyor (tab → dock title bar → panel
  içi büyük başlık). Panel'ler hem dock hem standalone dialog olarak kullanılıyor (`as_dock` flag) —
  başlığı kaldırmak dialog modunu bozar; bu bir tasarım refactor'ü, polish kapsamı değil.

### Çok-ajanlı polish denetimi (ultracode) — 2026-07-10

Arayüzün TÜM yüzeyleri offscreen render edildi (4 mod, 9 dock, 4 dialog, 4 tema),
sonra 10 paralel ajan (5 görsel + 5 kod) her yüzeyi taradı; her bulgu adversaryel
DOĞRULAMADAN geçti (default-reject + "sarı tema kullanıcının kasıtlı seçimi, bug değil"
guard'ı). 56 ham → 38 doğrulanmış (8 medium, 30 low). find→verify→sentez.

**Düzeltilenler (görünür, doğrulandı):**
- **B-119** — Advanced grubu: (1) collapsed'ken boş çerçeve çiziyordu (her açılışta orphan
  frame), (2) HC temada checkbox indicator'ı görünmüyordu (QGroupBox::indicator stillenmemiş).
  Fix: `[collapsed]` property + repolish → çerçevesiz disclosure satırı; QGroupBox::indicator
  QSS'i (tüm temalarda görünür). Dark+HC doğrulandı.
- **B-120** — QPI batch tabloları (dock + dialog) Stretch ile başlıkları ortadan kırpıyordu
  ("DRY MASS"→"RY MASS", "OPD"→"IPD"). Fix: ResizeToContents + stretchLastSection. Başlıklar tam.
- **B-121** — recon/qpi/report hata toast'ları `toast(msg,"danger")` çağırıyordu ama ToastHost
  level'ları info/ok/warn/error; "danger" tanınmayıp mavi 'accent'e düşüyordu → hata toast'ları
  KIRMIZI değil maviydi. 6 çağrı "error"a çevrildi (denetim sadece report'u bulmuştu; grep ile
  recon+qpi de bulundu).
- **B-122** — AI health pill hep muted gri; connected/unavailable/checking renk ayırt etmiyordu.
  Fix: QLabel[role=ok|warn|danger] + `_set_health(text,role)` repolish helper'ı.
- **B-123** (batch) — dosya adı iki kez (header caption kaldırıldı, panelin "Hologram" grubu tek
  kaynak), "Timelapse"/"Time-lapse" → "Time-lapse", 📎 emoji kaldırıldı, QSplitter::handle stillendi
  (grid ayraçları), camera Stop → danger rolü, busy_label → muted rolü.
- **#5** — light temada beyaz "500 µm" ölçek yazısı okunmuyordu (beyaz-üstüne-açık) → koyu backing
  pill (viewport.py); her temada okunur.
- **#7** — Reconstruct kontrol dock'u scroll-wrap edilmemişti (uzun panel kısa pencerede alt
  butonları kırpıyordu, B-113 feature dock'ları için yapmıştı ama bunu atlamıştı) → QScrollArea.
- **#8** — surface_viewer GL arka planı hardcoded (20,20,20); 2D viewport'lar palette.view_bg
  kullanıyor → ctx.palette.view_bg (tema-duyarlı, fallback koyu).

**Reddedilen (doğrulamada elendi / ben doğruladım):**
- **#2** (spinbox okları recon'da "kopuk") — YANLIŞ POZİTİF. recon alanları qpi ile yapısal olarak
  aynı (düz QDoubleSpinBox + aynı global QSS); pikselleri kendim inceledim, oklar entegre. Ajan
  divider çizgisini "boşluk" sanmış. Reddedildi.

**Kapsam-dışı, RAPORLANDI (bilinçli ertelendi):**
- Heading redundancy (#10/#17/#21, LOW): dock title bar + panel içi büyük H1 aynı adı tekrarlıyor.
  MİMARİ: panel'ler hem dock hem standalone dialog olarak kullanılıyor (`as_dock`), dialog modunda
  H1 tek başlık. Global kaldırmak dialog'ları bozar; dock-modu-özel bir heading-gizle seam'i gerek.
  Kullanıcı kararı gereken bir tasarım tercihi — polish kapsamında tek taraflı yapmadım.
- Token/tutarlılık mikro-item'ları (#24 min-height 22vs20, #25 eyebrow 3 farklı, #26/#31 margin=10,
  #28 form spacing, #34 label hizası, #35 pointSize): görünür etkisi ~sıfır, layout-shift riski var.
  Ayrı bir kod-tutarlılık turu için erteledim.
- #12 (form label L/R hizası), #13 (enum "LAPLACIAN_VARIANCE" ham gösterim), #15 (AI endpoint alanı
  URL'yi soldan kırpıyor), #19 (device Open/closed büyük-küçük), #20 (Z-ekseni boş grid hücresi),
  #32 (device section pattern): minör/subjektif/orta-risk, ertelendi.
- #14 (toast köşede kırpık): offscreen animasyon artefaktı (grab animasyon ortasını yakaladı), gerçek değil.

**Testler:** full suite 1260 passed / 11 skipped / 0 fail. 6 yeni polish testi (B-119..123) + B-118.
Registry 118→123.

### Ertelenen 2 follow-up tamamlandı — 2026-07-11

Kullanıcı iki task-chip prompt'unu yapıştırdı → ikisini de bu oturumda yaptım (ayrı session değil).

- **B-125 — panel adı tekrarı (dedup):** her feature panel adını 3 kez gösteriyordu (tab + dock
  title bar + büyük H1 heading). Araştırma: HİÇBİR panel as_dock=False mount edilmiyor, `_panel_action`
  docked panelleri dock olarak açıyor → hepsinin dock title bar'ı var. Fix: `mount_panel` +
  `_build_control_dock` → `_hide_panel_heading(widget)` (tek role="heading" label'ı gizler). Title
  bar/tab tek etiket; panel başına ~40px dikey alan geri kazanıldı. Standalone QDialog'lar
  (surface/qpi_batch/audit) mount_panel'den geçmediği için başlıklarını KORUYOR. Test + doğrulandı
  (recon/ai/qpi dock'ta heading yok; qpi_batch dialog'ta "QPI batch…" duruyor).
- **B-126 — magic-number tokenizasyonu:** (a) input min-height 22 / button 20 → ikisi Space.xl (24);
  (b) viewport başlık `setPointSize(10)` → ölçtüm (10pt==13px) → `setPixelSize(Type.label)` (boyut
  korundu); (c) camera/focus root margin 10 → Space.md (12, recon konvansiyonu); (d) camera/focus
  form label sağ-hizası override'ı kaldırıldı → hepsi platform-default (mac+offscreen tutarlı).
  **Alınmayan 2 madde (gerekçeyle):** #28 (form spacing 6→8) uzun kontrol panelini ~50px büyütüp
  B-124 fit işini bozardı; #25 (3 farklı "eyebrow" QSS'i birleştir) — üç kullanım padding/border/weight
  olarak ayrışıyor, ortak kural fayda getirmeden dolaylılık ekler.
- Not: iki task-chip "started" işaretli olduğu için dismiss edilemedi (prompt'u buraya yapıştırmak
  başlatmış sayılıyor) — iş burada tamamlandı, ayrı session'a gerek yok.

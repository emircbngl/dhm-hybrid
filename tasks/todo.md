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
(oturum sonunda doldurulacak)

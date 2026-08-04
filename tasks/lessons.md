# Lessons — DHM Reconstruction (Hybrid)

Running log of course corrections. Every entry should name what went wrong, why, and what rule to apply next time.

---

## 2026-04-29 — Pipeline default'ları iki tarafta da aynı olmak zorunda

**Mistake**: `core/batch_renderer.py` `subtract_mean`'i `process_state.get("subtract_mean", False)` olarak okuyordu. v1 `process_tab.py:44` checkbox default'u `True`, v2 `ReconParams.subtract_mean` default'u `True`. Profil dosyası bu key'i tutmadığında (yeni profil, hızlı batch run, programatik test) batch sessizce **DC bias'ı çıkarmadan** demodüle ediyor; canlı reconstruction frame'inde `~%50 amplitude scale` farkı oluşuyor. Phase aynı kalıyor (off-axis demodülasyon scalar offset'e duyarsız) ama amplitude metrik / dry-mass / contrast hesapları yanlış. Sentetik reproducer (`scripts/check_batch_vs_v2.py`) Scenario 1'de `max|Δamp|=0.2543` gösterdi; explicit `True` set edilince `0.0000`.

**Why**: Iki ayrı pipeline'ın "missing key" davranışı farklı default'a bağlandığında, profil migrasyonu / hand-crafted state dict'leri sessizce iki path'i ayırır. Phase tarafı bozulmadığı için review sırasında gözden kaçar — sayısal regresyon test'i olmadan hiç yakalanmaz.

**Rule**: 
1. **Tek-source-of-truth default tablosu**: bir parametrenin default'u kanonik bir yerde tanımlanır (örn. v2 dataclass), her okuyucu oradan bakar veya tablonun kendisini import eder. Iki ayrı `dict.get(..., default)` yazıyorsan, default'lar aynı literal olmak ZORUNDA — yorumla gerekçelendirip pin'le.
2. **Parite testi yaz**: aynı girdiyle iki pipeline çalıştır, complex field'ları `np.testing.assert_allclose` ile karşılaştır (atol ≤ 1e-6). Bu test düştüğü an default kayması ortaya çıkar. Ayrıca **source-level pin testi** yaz: kaynak dosyayı oku, literal default'un beklenen değerde olduğunu assert et — kimse "merge'de düzeltirim" mantığıyla değiştiremez.

**Reference subtract follow-up**: aynı audit'te yakalandı — autofocus/sweep search'leri reference DIVISION'ı *uygulamadan* propagate ediyor, sonra "best Z" bulunduğunda `_apply_ref(propagate(..., best_z))` ile referenced field kaydediliyor. Metric un-referenced field üstünde optimum buluyor, kullanıcı referenced field görüyor → "best Z" kullanıcının tepedeki Z'siyle uyuşmuyor. Fix: `core.autofocus.search_classic` zaten `ref_field=` kabul ediyor; `core/batch_renderer.py` üç algoritma branch'ında bunu forward etmeliydi. v1 `_reconstruct_reference` zaten `_reference_fc` (pre-propagation) saklıyordu, ama batch'e yalnızca `_reference_complex` (post-propagation, sabit z) geçiyordu. Düzeltme: `_reference_fc`'i de cfg'ye yaz; batch'te `_apply_ref(z_m=...)` her çağrıda referansı hedef z'ye yeniden yayınlasın.

**Pin**: `tests/test_batch_v2_parity.py` — 8 test, hem numerik parite hem de source-level default + ref forwarding'i kilitler.

---

## 2026-04-29 — "Lazy import" yorumu ≠ lazy davranış

**Mistake (AI ekibinde, audit'te yakalandı)**: `core/ai/client.py:88` yorumu "Lazy-import requests so module import doesn't fail in the test environment when the dep isn't installed yet" diyordu. Gerçek kod `__init__`'in içinde `import requests` çağırıyordu — module-level değil ama eager. `LocalLLMClient(...)` çağrısı `requests` yokken hard-fail edip `AIPanel.__init__`'i kırıyor, dolayısıyla ana pencere açılamıyor.

**Rule**: "Lazy import" pattern'i iki adımdan oluşur:
1. Import'u method body'sinin içine taşı
2. **Method'u ilk kullanım anına kadar çağırma** — yani `__init__` içinde de değil

`__init__` instantiation noktasıdır; oradaki import "construction-time eager" demektir. Gerçek lazy:

```python
def __init__(self, ...):
    self._session = None  # placeholder
    
def _ensure_session(self):
    if self._session is not None:
        return
    import requests  # noqa: WPS433  ← ilk gerçek kullanımda
    self._session = requests.Session()
    
def chat(self, ...):
    self._ensure_session()
    return self._session.post(...)
```

**Bonus**: ImportError'u `LLMClientError("install foo for AI") from exc` olarak sar — kullanıcıya "AI unavailable, install foo" mesajı verir, traceback değil.

**Pin**: `B-044` (`tests/test_ai_client.py`).

---

## 2026-04-29 — Qt thread'ler arası dialog: QTimer.singleShot ≠ thread crossing

**Mistake (AI ekibinde, audit'te yakalandı)**: `AIPanel._confirm` worker thread'inden `QTimer.singleShot(0, _run)` çağırıyordu, sonra `done.wait(120)`. `_run` 120 saniye boyunca hiç fire etmiyordu — irreversible tool confirmation modal'ı yıllardır dead code'du.

**Why**: `QTimer.singleShot(msec, callable)` callable'ı *çağıran thread'in* event loop'una post eder. AI worker thread'inde Qt event loop yok (sadece GUI thread'de var). Timer hiç açılmıyor.

**Rule**: Qt'da bir thread'den başkasına UI iş postlamak için canonical yol `QMetaObject.invokeMethod`:

```python
from PySide6.QtCore import QMetaObject, Qt, Slot

class Panel(QWidget):
    @Slot()
    def _do_dialog(self):
        # this slot runs on the panel's thread (GUI),
        # regardless of who called invokeMethod
        ...
    
    def trigger_from_worker(self):
        QMetaObject.invokeMethod(
            self, "_do_dialog",
            Qt.ConnectionType.QueuedConnection,
        )
```

`Slot()` decorator zorunlu — invokeMethod sadece kayıtlı slot'ları görür. `BlockingQueuedConnection` worker'ı slot tamamlanana kadar bekletir; `QueuedConnection` non-blocking + worker `threading.Event` ile sonucu poll eder.

**Pin**: `B-045` (`tests/test_ai_panel.py`).

---

## 2026-04-28 — Paralel ekipten gelen schema değişikliği zincirinin kontrolü

**Mistake**: AI ekibi `core/settings_schema.py` SCHEMA_VERSION 11→12'ye bump ettiler (yeni `AIDefaults` dataclass + `AppSettings.ai`). Qt-side `settings_store.py`'da `_migrate_v11_to_v12` eklemişler ama JSON-side `ui2/state_store.py`'a v11→v12 eklemeyi atlamışlar. Mevcut state dump'larını okumaya çalışan testlerin **6 tanesi birden patladı** çünkü `_migrate` v11'de takıldı, fallback "schema_version=12 reset" defaultlara döndü ama `_hydrate` `ai` field'ını yine populate etmiyordu.

**İkinci bağ**: aynı zamanda kendi mode-tracking eklememdeki `_set_input_mode` defensive değildi (`self._input_mode` set edilmemişken `prev = self._input_mode` AttributeError). DhmApp `__new__` ile kurulan headless test'leri kırdı.

**Rule**: Şirket içi başka ekibin schema'ya bumped commit gönderdiğinde tek bir doğru migration'ın **iki tarafa da** girmesi şart:
- `src/gui/settings_store.py::_MIGRATIONS` (Qt frontend)
- `src/ui2/state_store.py::_MIGRATIONS` (Dear PyGui frontend, JSON)
- `_hydrate` yeni field'ı populate ediyor mu kontrol

Süreç: schema bump'tan sonra **`scripts/check_bugs.py`'ı koşturmak schema migration regression'ı 30 saniyede yakalar.** Bu sefer 22 PASS → 16 PASS, 6 FAIL ile uyardı; o sinyal olmasa pilot'a giderdi.

**Ayrıca**: DhmApp gibi büyük UI sınıflarında yeni bir instance attribute eklediğinde `__init__` dışındaki kod path'leri (test'ler `__new__`, factory pattern'leri) açıkta kalır. Defensive read pattern: `getattr(self, "_attr", default)` yerine bare attribute access her yerde olmamalı.

---

## 2026-04-27 — Her phase kendi bug regression tool'unu yaratır

**Mistake pattern kaçınıldı**: önce tek `scripts/check_bugs.py` yazmıştım — 35 bug entry, hepsi aynı çuvalda. Pilot demo'da Anna "v2.0.7 ne durumda?" dediği an:
- Bütün suite koşmak gerekiyor (8+ saniye)
- Hangi rakamların hangi phase'e ait olduğu net değil
- Demo akışı bozuluyor

**Rule**: Her shipped phase (sprint / version) **kendi `scripts/check_bugs_phase_<name>.py` wrapper'ına sahip olur**. Yapı:

* `scripts/bug_registry.py` — tek source of truth, `BugEntry` + `BUG_REGISTRY` + `Phase` enum.
* `scripts/_bug_runner.py` — paylaşılan runner / formatter.
* `scripts/check_bugs.py` — tüm phase, `--phase <name>` filter destekler.
* `scripts/check_bugs_phase_<name>.py` — 5-satırlık wrapper, `run_phase(Phase.<NAME>)` çağırır.

**Pratik fayda**:
- Pilot demo'da "v2.0.6 phase yeşil mi?" → 1 sn'lik komut, sadece o phase rakamları
- Future phase'ler için boş wrapper'lar önceden var → ritual day-1'den hazır
- `Phase` enum chronological → registry'den geçmiş kronolojik akıyor
- Yeni bug'a phase field zorunlu → "hangi sprint'te yakaladık" hep takipte

**Sprint cycle'a eklendi (`tasks/roadmap.md` sprint ritual section)**:
1. Sprint sonunda yeni bug'lar `bug_registry.py`'ye phase tag'iyle eklenir.
2. Yeni bir version başlarken, `Phase` enum'una entry + boş `check_bugs_phase_<key>.py` wrapper push edilir (henüz kayıt olmasa bile).
3. `python scripts/check_bugs_phase_<current>.py` her demo öncesi koşulur.

**Kontra-kural**: yeni phase wrapper yazmadan da çalışan tek genel tool olsun istiyorsan `scripts/check_bugs.py --phase <name>` aynı işi yapar. Wrapper'lar yalnızca operatör kolaylığı; registry tek doğrudur.

---

## 2026-04-24 — Tahmin yapma, ölç: `spectrum.copy()` perf tahmini 4200× kat şişikti

**Mistake**: `find_focus_candidates`'i `_make_fast_evaluator`'a refactor ederken yorumuna "25-30 % wall-clock saving on 1024² × 60-step" yazdım — `propagate()` path'indeki `spectrum.copy()`'nin ağır olduğunu varsaydım.

**Gerçek**: ölçtükten sonra 1594 ms → 1598 ms geldi — **%0 kazanç**. Tahminim yanlıştı çünkü complex64 @ 1024² = 8 MB, modern Mac'te 50 GB/s memory bandwidth ile 0.16 ms per copy. 60 call = 10 ms = toplam runtime'ın %0.6'sı. FFT + metric compute dominant, memcpy değil.

**Rule**: Performans yorumuna rakam yazacaksan önce ölç. "Should be fast because X" yorumları refactor PR'larını kirletir ve gelecekteki kendinin hayatını zorlaştırır — commit'e rakam koydun mu, rakam yanlışsa commit yanlış. Yazmak 30 sn; doğru yazmak için bench 2 dk. İkisini de yap.

**Rule sonucu**: refactor kalsın çünkü code quality win (tek path, `_GLOBAL_RECON_CACHE` dependency kalktı), ama yorum güncellendi: "perf-wise this refactor is a wash" dedim, somut rakam yazdım. Gelecek refactor'lar `_make_fast_evaluator`'a win eklerse multi-focus onu otomatik alır — asıl value bu, ölçülmeden tahmin edilen "25 %" değil.

---

## 2026-04-24 — "Autofocus yavaşladı" raporu önce ölçülmeli, sonra çözülmeli

**Mistake pattern kaçınıldı**: user "önceden 5 sn'de buluyordu, şimdi yavaş" dedi. Refleks "optimizasyonu tekrar yaz" demek olurdu. Onun yerine önce benchmark yazıldı — 6 algoritma × 4 boyut (256, 512, 1024, 2048):

| shape | zscan | coarse_to_fine | robust | adaptive_gradient | adaptive_bracketing | adaptive_distance |
|-------|------:|---------------:|-------:|------------------:|--------------------:|------------------:|
| 256²  |   65  |    95          |   120  |        66         |      88             |     23            |
| 512²  |  250  |   360          |   460  |       255         |     340             |    240            |
| 1024² | 1070  |  1530          |  1920  |      1080         |    1430             |    990            |
| 2048² | 4280  |  6170          |  7710  |      4300         |    5730             |   4060            |

Dispatcher overhead (TIFF load + preprocess + extract) 512² için sadece **14 ms** — core'a karşı pratik olarak sıfır. Yani "core yavaş değil" kesin.

**User'ın "5 saniye" anısı**: 2048×2048 × 40 step zscan = 4.3 sec. Tam eşleşme. Büyük hologramda zscan zaten ~5 saniye; "yavaşladı" algısı muhtemelen farklı algoritma seçimi (robust: 7.7 sec, coarse_to_fine: 6.2 sec @ 2048²) veya daha yüksek `n_steps` yüzünden. Algoritma × boyut matrisini görmeden "fix" yazmak kör olurdu.

**Rule**: Performans şikâyetinde ilk adım **her zaman** benchmark. "Optimize edildi sanmıştık" demeden önce *bugünkü* timing'leri ölç, user'ın gerçek senaryosunu (shape + algorithm + n_steps + TIFF I/O) iste, ikisinin çarpımında nereye düştüğüne bak. Aksi halde var olmayan bir regresyonu kovalarsın.

**Pin**: `tests/test_autofocus_speed_baseline.py` — 256/512/1024 için zscan ceiling'leri 2× mevcut sayılarla pin'lendi. Gelecekte `_make_fast_evaluator`'daki FFT cache'i breakler, CI'da görürüz.

**Benchmark script**: `scripts/bench_autofocus.py` — `--shapes 256,512,1024,2048` argümanı ile çalışır, JSON + markdown tablo yazar. Yeni algoritma eklendiğinde / FFT backend değiştiğinde direkt yeniden çalıştır.

---

## 2026-04-24 — Single-sphere synthetic lateral diameter is not measurable post-reconstruction

**Mistake**: Tried to build an end-to-end test that runs autofocus then measures lateral sphere diameter on a single-sphere synthetic hologram with Δn = 0.07. Every diameter metric (FWHM of centre row, area-above-threshold equivalent disk, phase-std at sphere vs background) reported nonsense — 4–5× truth, or no discrimination at all (ratio ≈ 1 between sphere patch and corner patch).

**Root cause**: Two stacking effects on single-sphere synthetic scenes:
1. Δn = 0.07 over a 15 µm sphere creates phase that wraps 3–4 times; any area-above-threshold metric picks up wrap-induced sin() lobes + Fresnel ringing rather than the physical boundary.
2. Off-axis extraction leaves a residual carrier beat (period ~4 px here) that fills the entire frame at roughly uniform amplitude — so local statistics (std, variance) inside any window are similar regardless of position.

In the existing `test_triple_sphere_lateral_diameter_within_tolerance` the same diameter metric *does* work because the multi-object scene creates localised contrast (the other two defocused spheres act as a structured background). Remove one sphere and the signal disappears into the carrier.

**Rule**: For end-to-end synthetic validation "after autofocus":
- **Depth correction** → assert z_est within 5–7× scan step of truth. Reliable.
- **Lateral presence** → assert OPD peak-to-peak inside a *tight disk at the sphere's expected pixel* reaches min(2·r·Δn, λ) within ±70 %. The tight ROI doubles as a lateral-position check — a sphere landed at the wrong pixel would give an all-background ROI with much smaller swing.
- **Don't assert a scalar diameter** from a single-sphere synthetic reconstruction; that question needs a multi-object scene, or a lab hologram, or both.

Also: don't reach for a complex-field similarity metric as a cheap substitute. At the autofocus tolerance boundary (5× step) the similarity drops to ~0.5 legitimately (Fresnel kernels at 4× step differ enough to decorrelate), so the threshold that would accept autofocus's own output also accepts half of "broken pipeline". The OPD-in-disk check is the tightest honest constraint.

---

## 2026-04-17 — Autofocus direction inferred from synthetic ≠ real

**Mistake**: Re-flagged TOTAL_VARIATION, GRADIENT, TENENGRAD, LAPLACIAN_VARIANCE, BRENNER, SPECTRAL_ENERGY as *minimize-at-focus* after watching a synthetic test case (a Gaussian phase bump) and reading `test_phase_af.py`'s Fresnel-ring-based reasoning. Real lab hologram then converged to z ≈ 117 mm instead of the landscape peak at ≈ 45 mm.

**Root cause**: Synthetic "phase bump on flat background" is dominated by defocus-induced Fresnel rings that *add* high-frequency content away from focus. Real DHM samples (cells, beads, stages) have sharp in-focus structure that dominates; at defocus, structure smears and gradients DROP. The directions flip between these two regimes.

**Rule**: Never set `_is_minimize` direction from synthetic reasoning alone. Always confirm on a real hologram with a full landscape scan — if the metric's global peak on the real scan sits at the same z where reconstruction looks focused to the eye, it's a *maximize* metric.

**Current direction convention**: only `ENTROPY` minimizes. Every other phase metric maximizes at focus.

---

## 2026-04-17 — Wrong venv when multiple virtualenvs exist

**Mistake**: Ran `./venv/bin/pip install PyOpenGL ...` from `Hybrid/`, but the shebang in that `pip` was a leftover hardcoded reference to `../Phyton/venv/bin/python`, so the package installed into the wrong env and `import pyqtgraph.opengl` still failed.

**Rule**: When a project lives next to sibling envs (`Hybrid/venv`, `Phyton/venv`, etc.), invoke pip as a module: `venv/bin/python -m pip install …`. That resolves the interpreter from the env's own binary instead of trusting a possibly-stale shebang.

---

## 2026-04-17 — Silent-import warnings swallowed in Qt code

**Mistake**: `_show_3d_surface` caught ImportError on `pyqtgraph.opengl`, logged a warning, and returned. User-facing result: the button did nothing, with no feedback. Diagnosis took far longer than it should have.

**Rule**: In GUI paths, when a missing-optional-dependency error prevents an action, *both* log it AND push a user-visible message (`self.status_bar.show_message(...)` in this codebase). Silent warnings in a Qt app are effectively swallowed — the log pane is usually not open.

---

## 2026-04-17 — Wrapped-phase variance explodes near ±π

**Mistake**: `PHASE_VARIANCE` computed `np.var(wrapped_phase)`. On real holograms the wrap boundary makes the histogram look bimodal at ±π, so the metric peaked at heavy defocus (z≈70mm) instead of the real focus (z≈45mm). Caused AdGrad to converge to the wrong z.

**Rule**: For *any* statistic on wrapped phase, decompose to sin/cos or use circular statistics. The safe variance is `1 − |mean(exp(iφ))|` (circular variance). Same rule applies to gradients (use `_wrap_diff`) and Laplacians (apply to sin/cos separately).

---

## 2026-04-17 — Greedy walker + wide range = local-max trap

**Mistake**: `adaptive_gradient_search` Phase 1 walker shrunk its step on the first rising shoulder and never reached the global peak. Budget exhausted before traversing the full range, so Phase 2 refined around a wrong best_z and couldn't escape.

**Root causes (compounded)**:
1. `prev_deriv = 0.0` initial value tripped the "derivative grew" branch on the very first iteration → step shrank before the walker moved.
2. No coverage check — if the walker only covered 60% of [z_min, z_max], Phase 2's local refinement couldn't rescue.
3. `ad_budget = max(30, steps * 0.5)` in the worker couldn't reach ±50mm from ±0.5mm (log₂(100) ≈ 7 expansions × 15 pts = 105 evals needed).
4. Worker forced `step_init = (zmax-zmin)/20` after AdDist, which made the walker take 20 tiny steps to traverse instead of using the algorithm's natural `full_range/10` default.

**Rule**: A greedy local-search walker must have either (a) a guaranteed-coverage fallback (uniform sweep if the walker didn't traverse the range), or (b) a budget-aware step floor that forces traversal. Pre-validation belongs in the algorithm — don't trust an outer pipeline to supply "good enough" ranges.

**Rule (worker sizing)**: When an expanding-discovery phase feeds downstream refinement, budget it by `ceil(log2(max_range / init_range))` iterations, not a flat `max(30, steps/2)`. That turns the budget into a property of the problem instead of a property of the UI.

---

## 2026-04-17 — Know when NOT to refactor

**Context**: Plan called for splitting `main_window.py` (2653 lines) into mixin classes. Started mapping the sections; backed off before writing code.

**Rule**: Mixin-splitting a single Qt `QMainWindow` class trades a readability win for worse tooling (type checkers can't track cross-mixin `self.x`, IDE jump-to-def breaks, subtle MRO hazards). Only do it when the file is *actively painful to edit*, not merely large. For GUIs, prefer extracting *standalone* subwidgets / free-function helpers into `src/gui/views/` rather than horizontally slicing one class. Independent-function modules (like `src/core/autofocus.py`) split cleanly and are the right candidates; stateful Qt classes usually aren't.

---

## 2026-04-17 — Async `destroyed` signal killed the replacement window

**Mistake**: `_show_qpi_window` connected `dlg.destroyed → _cleanup_3d_window`. When a new reconstruction auto-refreshed QPI, the old dialog was closed (WA_DeleteOnClose) and a new one opened immediately. The old dialog's `destroyed` signal fires *asynchronously* after the new 3D window already exists, so `_cleanup_3d_window()` ran on the fresh widget and tore it down.

**Rule**: When you have a parent→child cleanup wired through a `destroyed` signal, *disconnect it* before you replace the parent programmatically. `WA_DeleteOnClose` means close is not destroy — the destroy fires later, after whatever new object you've created in between. Pattern:
```python
try:
    dlg.destroyed.disconnect(self._cleanup_3d_window)
except (TypeError, RuntimeError):
    pass
dlg.close()
```

---

## 2026-04-17 — Destroying a GLViewWidget and rebuilding it renders black

**Mistake**: `_show_3d_surface` used to `deleteLater()` the old `GLViewWidget` and create a fresh one every call. On the second open the new widget rendered pitch-black even though the data was identical. Camera, grid, items all present — just no visible draw.

**Root cause**: `deleteLater` schedules an async destroy. The OpenGL context teardown for the *old* widget happened on the same GUI tick as the `initializeGL` call for the *new* widget, and Qt didn't give the new context a clean slate.

**Rule**: Prefer *reusing* a `GLViewWidget` (or any QOpenGLWidget) across refreshes. Check `shiboken6.isValid(w)` to tell whether the Python handle still points at a live C++ object; if yes, call `w.removeItem(item)` for each item in `w.items` and re-add fresh items. Only create a new widget if the old one is truly dead (user closed via X with WA_DeleteOnClose, mode switch cleaned it up, etc.). Camera position should only be set on *first* open so the user's rotation survives refresh.

**Tangential rule**: For widgets meant to be long-lived (progressive refinement, live refresh), don't set `WA_DeleteOnClose`. Let `hide()` + reshow handle the open/close cycle; reserve deletion for mode changes and file switches.

---

## 2026-04-20 — `git add -A` pulled in 56k lines of worktree backups

**Mistake**: Asked to save a snapshot, ran `git add -A` after only seeing the first 30 lines of `git status --short` (piped through `head -30`). That sample hid untracked directories — including `.claude/worktrees/v1.0_apr03_base/` (a full copy of an old version) and macOS `.DS_Store` files. Committed 561 files / +56k lines instead of the ~10 real changes.

**Root cause**: Truncated status output + blind wildcard stage. Never enumerated what `-A` would actually add.

**Rule**: Before any `git add -A` or `git add .` on a dirty repo:
1. Run `git status --short` *unfiltered* (or pipe to `| wc -l` first to gauge size).
2. Scan untracked section for unexpected top-level directories (`.claude/`, `worktrees/`, `node_modules/`, anything macOS/IDE-specific).
3. Ensure `.gitignore` covers local-only state *before* staging, not after.
4. Prefer `git add <explicit paths>` when the modified set is small and known.

**Recovery pattern used (non-destructive)**: `git tag -d <tag>` → `git reset --mixed HEAD~1` → fix `.gitignore` → re-stage → re-commit → re-tag. No force-push, no `reset --hard`, working tree preserved.

---

## 2026-04-21 — `QShortcut` moved to `QtGui` in PySide6 6.x; unit tests don't catch it

**Mistake**: `main_window._setup_shortcuts` and `gui/commands_install.install_shortcuts` both imported `QShortcut` from `PySide6.QtWidgets`. All 137 pytest cases passed — they exercised the command *registry* but not the QShortcut *binding* path, because no test instantiated a full `MainWindow`. Manual smoke (`MainWindow()` under offscreen Qt) immediately raised `ImportError: cannot import name 'QShortcut' from PySide6.QtWidgets`.

**Root cause**: PySide6 6.0 moved `QShortcut` from `QtWidgets` to `QtGui`. Widget-level tests that only touch individual widgets never trigger the failing import; only a full main-window boot does.

**Rule**: For any task that touches `main_window.py`, include a headless-Qt smoke that actually runs `MainWindow()` to construction completion. A 5-line smoke (`QApplication([]); MainWindow()`) catches whole classes of "import-at-first-use" regressions that unit tests miss. Add the smoke to the verification checklist, not just the unit tests.

**Qt version note**: for PySide6 6.x use `from PySide6.QtGui import QShortcut, QKeySequence, QAction`. Pinning the import at the Qt-moved site (not at module top-level) limits the blast radius.

---

## 2026-04-21 — `np.bool_` ≠ `bool` in `is` checks

**Mistake**: `has_sufficient_contrast()` in `src/core/autofocus/metrics.py` returned `circ_var >= min_circular_std`, which is a `numpy.bool_` (not Python `bool`). Two pytest cases used `assert has_sufficient_contrast(...) is False` / `is True` — identity check against Python singletons — and failed because `np.False_ is False` → `False`.

**Root cause**: The function's type hint said `-> bool` but the body leaked a numpy scalar. Test authors trusted the annotation and used identity comparison (which is idiomatic and correct for Python booleans).

**Rule**: When a function's return annotation is `bool`, make it a real Python `bool` at the boundary: `return bool(expr)`. Don't let numpy scalars escape typed APIs — they look equal but fail `is` checks and pickle oddly. Same rule applies to `int` / `float` annotations returning `np.int64` / `np.float64`.

---

## 2026-04-24 — Dear PyGui `set_viewport_drop_callback` sessizce macOS'ta ölü

**Mistake**: v2.0.1'de viewport-level drop callback'i `try/except` ile sardık, "OS older builds" için yedek. Gerçek problem: Dear PyGui 2.x macOS'ta sembolü ilan ediyor ama Cocoa binding'i drop event'ini hiç emit etmiyor. Kullanıcı status bar "Drag a hologram onto the window to load it" gördü, denedi, hiçbir şey olmadı, hata bile yok. Kör bir promise.

**Root cause**: Platform-specific capability'yi "try bind — fail silently" modeliyle handle ettik. Bu model sadece API yok olduğunda çalışır; API var ama davranış yok olduğunda (macOS Cocoa burada) kullanıcıya yalan söyleriz.

**Rule**: Platform-conditional GUI feature'ları için "capability detect → honest degrade" modeli. `sys.platform` check ve gerekirse `DHM_FORCE_DROP` tarzı env override (upstream fix gelirse kullanıcı kendi aktif edebilir). Status bar / tooltip / onboarding metinleri *her zaman* gerçek kapasiteyi yansıtsın — "supported" dediği yerde çalışsın, çalışmıyorsa "click here instead" gibi alternatif affordance göster. Sessiz `try/except` + debug log = kullanıcı için = çalışmıyor + hata yok.

---

## 2026-04-24 — Dear PyGui viewport sizing sabit ≠ her ekran

**Mistake**: v2'de `DEFAULT_SIZE = (1280, 800)` sabitledik, içerik 320 sidebar + 2×512 panel + padding > 1360px yatay ve `content_row` üzerinde `no_scrollbar=True` verdiğimiz için MacBook 13"/1366×768 pilotunda içerik kesildi, scroll bile yoktu.

**Root cause**: (a) İçeriğin minimum genişliğini hesaplamadan viewport'a küçük default verdik, (b) scroll sigortasını kapattık — layout hatası görünmez oldu.

**Rule**: GUI shell açılış boyutu iki yönden türetilmeli: (1) içeriğin minimum invariant'ı (`sidebar + 2*(preview+pad)` gibi hesaplanabilir bir lower bound), (2) ekranın gerçek boyutu (tkinter'ın `Tk().winfo_screenwidth()` stdlib probe'u yeterli, withdraw + destroy ile flicker-free). Preview/panel boyutunu viewport genişliğine göre tier'la (288/384/512 gibi) — tek magic number değil bir fonksiyon. Scroll sigortasını *açık* bırak — responsive tier logic çalışsa bile bir edge case scroll tetikleyecekse, içerik kesilmesin.

---

## 2026-04-24 — Info text append birikince stale state gösterir

**Mistake**: v2.0.1'de `_handle_qpi` ve `_handle_depth` info panel'e `dpg.get_value("info_text") + "\n\n" + msg` yazdı. QPI sonrası depth çalıştırınca metin uzadı. `_clear_depth_overlay` depth tint'i silse bile info metindeki "Depth map: …" satırı orada kaldı.

**Root cause**: UI metninde append = stale state kaynağı. Text widget'ı sanki log buffer gibi kullandık, halbuki aslında "şu an neye bakıyoruz" özeti.

**Rule**: UI metnini *cache'ten* compose et, append etme. Yani: `_last_recon`, `_last_qpi`, `_last_depth` referans tutan bir `_compose_info_text()` fonksiyonu yaz; her state değiştiğinde `_refresh_info_text()` çağır. Append pattern sadece append-only-log widget'ları (gerçek terminal/transcript) için uygun. Status/info/dashboard text'i replace-only olmalı ki state kirlenmesin.

---

## 2026-04-24 — v1→v2 frontend portunda bilimsel parametre audit yapmadan geçme

**Mistake**: v2 (Dear PyGui) frontend'ini yazarken v1 `ReconTab`/`AutofocusTab`/`QPITab`'tan ancak **bir kısım** parametreyi taşıdık. Kaçanlar: `magnification`, `pixel_is_effective`, `n_sample`, `n_medium`, `autofocus_metric` (11'den 1'ine indirgenmişti), `z_min/z_max` user-configurable, `subtract_mean`, FFT backend seçimi, TIFF metadata auto-detect. Sonuç: **40× mikroskop setup'ında z propagation M² oranında kayar** — 10 mm yerine 16 m. QPI dry mass yanlış `n` ile hesaplanır. Autofocus low-contrast bio örneklerde fail eder. Kullanıcı pilot testte "magnification neden yok?" sorduğunda fark ettik — tek bir soru 9 ayrı bilimsel gap'i ortaya çıkardı.

**Root cause**: UX portunu "ekranları yeniden çiz" olarak yaptık, bilimsel parametreleri "hardcoded default yeterli" varsayımıyla geçtik. Kullanıcı-editable parametreleri v1'deki tab'lara bakmadan, `ReconParams`'ı minimal tutarak başladık — ama minimal != doğru. Minimal, sessiz bilimsel yanlışlıktı.

**Rule**: Yeni frontend yazarken ilk iş v1'in **tüm** parametre surface'ini audit et. Her sidebar tab'ındaki her widget listelensin; `main_window.py` grep'iyle `rtab.X.value()`, `ftab.X.value()`, `qtab.X.value()` çağrılarının hepsi bulunsun. Her parametre için soru: (a) fiziksel sonucu etkiliyor mu, (b) örnek-bağımlı mı, (c) auto-detect edilebilir mi? Cevaplardan biri "evet" ise v2'ye **mutlaka** taşı. "Default yeterli" argümanı sadece c=false ve a=false için geçerli — bilimsel parametrelerde asla. Ayrıca: yeni bir fizik-bağımlı field eklendiğinde `SCHEMA_VERSION` bump + migration'da **defaults-match-previous-behaviour** backfill yap, böylece eski state dump'ları yeni kod üzerinde bit-identik davranır; behaviour değişimi sadece kullanıcı widget'a dokununca olur.

---

## 2026-04-24 — Paket __init__.py'ın eager import'u headless testi kırdı

**Mistake**: `src/ui2/__init__.py` top-level'de `from .app import DhmApp` yapıyordu. Ama `ui2/app.py` → `import dearpygui.dearpygui as dpg` + `ui2/theme.py` da top-level'de aynı import. Bu yüzden `from ui2.state_store import load` gibi **GL-free** bir import bile Dear PyGui'nin C extension'ını yüklüyordu. Testlerde `sys.modules["dearpygui"]` stub'lamak istediğimde real library zaten yüklü olduğu için stub atlanıyor, sonraki dpg çağrılarında segfault oluyordu.

**Root cause**: Paket __init__.py'ı "public API'yı expose et" için eager import kullandım, ama bu API'nın bir kısmı (DhmApp) GL backend gerektiriyor. Diğer alt modüller (state_store, reconstruction dataclasses, workers driver class'ları) tamamen headless olmasına rağmen eager import onları da GL'e bağımlı kılıyor.

**Rule**: Opsiyonel/ağır bağımlılığı olan public sembolleri paket __init__.py'da **PEP 562 lazy attribute** ile expose et:
```python
def __getattr__(name: str):
    if name == "DhmApp":
        from .app import DhmApp as _DhmApp
        return _DhmApp
    raise AttributeError(...)
```
Ayrıca en alt-modüllerde de (örn. `theme.py`) ağır import'u class metoduna kaydır (`def apply(cls): import dearpygui.dearpygui as dpg; ...`). Böylece `import ui2.state_store` sadece core'a bağlı olur, GL backend'i dokunulmaz. Testlerde stub atarken "dearpygui" ın sys.modules'a pre-load'u bir trap; `ui2` paketini import eden herkes unintended eager side-effect'ler olmadığından emin olsun.

---

## 2026-04-24 — Core scan fonksiyonuna cancel_check optional parametresi eklemek

**Mistake**: v2.0.2'de Esc sadece status bar'ı "Ready." yapıyordu. Arkada autofocus/multifocus/depth scan'leri 2-10 saniye çalışmaya devam ediyordu çünkü core fonksiyonları cancel mekanizması kabul etmiyordu. Thread'i sonlandırmaya çalışmak Python'da temiz değil (signal + FFT ortasında crash riski).

**Root cause**: Uzun-running core fonksiyonları "atomik" davranıyordu — içi scan loop'u olmasına rağmen dışarıdan durdurulamıyordu. UI layer'ı "cancelled" diye bildirim atıp result'ı atsa bile CPU boşa gidiyor, kullanıcı bir sonraki işi başlatmak için bekliyor.

**Rule**: Long-running core fonksiyonlarının **her scan iterasyonunda** opsiyonel bir `cancel_check: Optional[Callable[[], bool]] = None` parametresini çağır. `True` dönerse domain-spesifik cancellation exception raise et (bu projede `AutofocusCancelled`). Optional default=None backward compat garantisi. UI layer `threading.Event` tutar + callback `lambda: event.is_set()` verir; Esc event'i set eder, core boundary'de observe edip raise eder. Exception UI'da `on_error("Cancelled.")` şeklinde sinyallenir. Bu cooperative cancellation pattern'ı Qt'nin `QThread.requestInterruption()` + `isInterruptionRequested()` semantiğinin pure-Python karşılığı.

---

## 2026-04-24 — `find_focus_candidates` top candidate true z'den 2-3 scan step sapabiliyor

**Discovery**: Multi-focus validation testi yazdım, tek sphere @ z=10mm üzerinde top-ranked candidate 9.48mm (0.52 mm off, 2.5 scan step). z=15 ve z=20mm'de sorunsuz, sadece bazı z'lerde systematic offset. Sebep: sphere'in Fresnel-ring envelope'u metrik peak'ini biraz kaydırıyor — primary peak ile envelope arasında prominence savaşı.

**Rule**: Multi-focus feature'ının kullanıcı-facing accuracy'si scan step'inin ~2-3 katı — bu kabul edilebilir çünkü (a) kullanıcı candidate listesinden seçer (primary değilse ikinciyi dener), (b) sonra reconstruction z slider'ıyla ince ayar yapar. Test assertion'ları "top-1 = truth ± step" yerine "top-5 içinde ≥1 entry truth ± 4*step" olarak formüle edilmeli — bu gerçek kullanım modelini yansıtıyor. Daha sıkı assertion bir accuracy yalanı.

---

## 2026-04-24 — Dear PyGui stub top-up across test files

**Mistake**: v2.0.6 test dosyalarının her biri kendi içinde `_install_dpg_stub()` çağırıyor ama "if 'dearpygui' in sys.modules: return" guard'ı daha önceki test file tarafından install edilen stub'ın bu test için gereken ek attr'ları (örn. `get_plot_mouse_pos`) taşımadığı durumda attributeerror'a düşüyor. 2 test cross-file ordering sensitivity nedeniyle fail etti (full suite: line_profile gesture testleri).

**Root cause**: Her test file kendi için guard ediyor ama guard "stub zaten yüklü, devam et" değil, "hiçbir şey yapma" demek. Yeni test yazarken stub'a eklediğim attr'lar önceki test'in stub'ına hiç düşmedi.

**Rule**: Stub installer her zaman **idempotent + top-up** ol: module zaten `sys.modules`'taysa `hasattr` ile eksik attr'ları ekle, `return` etme. Alternative: conftest.py'da merkezi stub + per-test fresh copy, ama bu daha büyük refactor. Minimum fix: her test file'ın stub'ında `if existing: merge missing attrs; return` pattern'ı. `AttributeError: <module 'dearpygui.dearpygui'> has no attribute 'X'` görürsen cross-file stub eksiği demektir — o attr'ı her stub installer'ın top-up block'una ekle.

---

## 2026-04-24 — HDF5 group name collision + metadata stringify

**Mistake**: Batch bundle writer ilk versiyonunda `foo.tif` ve `foo.png` aynı grup key'ine düşüyordu — h5py ikincisini ilkinin üstüne yazmak yerine duplicate key error veriyor ya da ilki kaybolmuş gibi görünüyor. Ayrıca metadata attr'a Path objesi yazmaya çalışınca `TypeError: Object dtype dtype('O') has no native HDF5 equivalent`.

**Root cause**: Dataset/group names HDF5'te string olmalı + unique olmalı; metadata attr'larına yazılacak değerler scalar (int/float/bool/bytes) veya short string olmalı. Stem-based naming batch için doğrudan güvenli değil.

**Rule**: HDF5 group naming'inde her zaman `_safe_key(stem, taken_set)` pattern'ı — slug `[^A-Za-z0-9_\-]+ → "_"` + `_02`, `_03` suffix çakışma durumunda. Metadata attr write'ında `isinstance(v, (str, bytes, int, float, bool, np.integer, np.floating))` whitelist; dışında kalan her şey `str(v)` ile stringify. Bu bir silent data change değil — metadata zaten "doc string" kategorisinde, Path objesi roundtrip ettikten sonra `str` olarak geliyor, semantic loss yok. Yazıcıya "her şey stringa dönebilir" garantisi verirsen batch mid-way TypeError'la patlamaz.

---

## 2026-04-24 — Daemon thread stop latency bütçesi

**Mistake**: İlk AcquisitionThread implementasyonu `time.sleep(interval)` ile uyuyordu — stop() çağrısı ile thread'in exit etmesi arasında full interval (33 ms @ 30 fps) beklemeye neden oluyordu. Düşük FPS ayarlarında (örn. 1 fps) stop latency 1 saniyeye kadar çıkıyor, UI "stop butonuna bastı, hala çalışıyor" hissi veriyordu.

**Root cause**: Uzun uninterruptible sleep = yüksek stop latency. Event.wait gibi interruptible primitive'lere pass etmek gerek ama bu durumda FPS timer ile çakışıyor.

**Rule**: Daemon worker thread'lerde sleep chunk'larına böl: `while sleep_for > 0 and not stop_event.is_set(): time.sleep(min(sleep_for, 0.1)); sleep_for -= chunk`. Bu sayede stop latency ≤ 100 ms; user perceives instant. FPS timing doğruluğu için `next_tick += interval` pattern'ı — drift'i absorb eder, her iterasyon mutlak zamanla hizalanır.

---

## 2026-04-24 — `mvClickedHandler` Dear PyGui'de her widget type'ta geçerli değil

**Mistake**: v2.0.1'de drop-zone için `dpg.child_window(tag="drop_zone")` + `item_handler_registry` + `add_item_clicked_handler` kombinasyonu kullandım. Pytest'te stub MagicMock no-op olduğu için test yeşil geçti, ama gerçek app startup'ta Dear PyGui `Item Handler Registry includes inapplicable handler: mvClickedHandler` hatasıyla crash'ledi. Drop zone click-to-browse hiç çalışmamıştı; sadece **crash handler + launcher probe** v2.0.6'da devreye girince görebildik.

**Root cause**: Dear PyGui'de `mvClickedHandler` (add_item_clicked_handler) sadece **butonlar, checkbox'lar, sliders** gibi "activatable" widget'larda çalışıyor — `mvChildWindow` bunu kabul etmiyor. Stub'lar bunu test etmiyor çünkü hiçbir widget type check'i yapmıyorlar; sadece real Dear PyGui runtime'da ortaya çıkıyor. Bu da "stub green != production green" lesson'ını tekrar öğretiyor.

**Rule**: Container widget'larında (child_window, group, window, table_row) click handler gerekiyorsa:
- **Opsiyon A**: Widget'ı `dpg.add_button(...)` ile değiştir — tam satır genişliğinde styled. Buttons mvClickedHandler'ı kabul eder.
- **Opsiyon B**: Global `handler_registry()` içinde `dpg.add_mouse_click_handler(button=mvMouseButton_Left, callback=...)` ekle + callback'te `dpg.is_item_hovered("container_tag")` ile gate yap.
Opsiyon B container'ı göstermeye devam etmene izin verir, A'dan daha esnek. `item_handler_registry` yerine global `handler_registry` kullan. Gelecekteki testler için: smoke test olarak `subprocess` ile launcher'ı 2-3 saniye çalıştırıp crash dump dizinini kontrol et — MagicMock testleri gerçek Dear PyGui C extension'ının widget-type rule'larını bilmiyor.

---

## 2026-04-24 — Sibling venv ladder her zaman `-x` kontrol etmiyor olabilir — capability probe

**Mistake**: `Run v2.command` ilk versiyonu "eğer venv/bin/python executable ise onu kullan" ladder'ına güveniyordu. `Hybrid/venv` var ve executable ama **dearpygui yüklü değil**; ladder onu seçince `dearpygui is required for the v2 UI` mesajı + exit. Kullanıcı Finder'dan launch etti, anladı ki "v2 açılmıyor". Silent misfire — ladder doğru çalışıyor ama yanlış interpreter'ı seçiyordu.

**Root cause**: Interpreter varlığı ≠ doğru runtime. Proje tree'sinde birden çok venv olabilir (Hybrid/venv = başka iş için kurulmuş minimal env, Phyton/venv = v2 için kurulmuş full env, sistem python = bağımlılık yok). "İlk executable" mantığı kullanıcı-friendly değil; kullanıcının venv düzenini bilmesini gerektirir.

**Rule**: Launcher'da capability probe kullan — `python -c "import <critical_module>"` komutunu her aday için çalıştır, ilk sıfır-exit olanı seç. Her başarısız olanı `PROBE_FAILED` array'ine biriktir ve hepsi fail ederse kullanıcıya "bu interpreter'lardan hiçbiri <module>'a sahip değil, `<suggested_interpreter> -m pip install <module>` ile yükle" mesajı ver. Minimum `set -u` güvenliği için Bash/zsh array pattern'ı `${array[@]}` içinde tırnak — her path'in boşluk/özel karakter içerebileceğini unutma. `echo "Using Python: $PY"` transparansı için önemli — user launcher'ın hangi interpreter'ı seçtiğini ekranda görebilmeli.

---

## 2026-04-24 — `viewport_menu_bar()` macOS'ta uygulama içinde görünmüyor

**Mistake**: v2 menü bar'ını `dpg.viewport_menu_bar()` kullanarak kurdum. Windows/Linux'ta menü pencerenin üstünde görünüyor, macOS'ta ise sistemin native menu bar'ına (ekranın en üstü) gidiyor. Kullanıcı "yukarıdaki Reconstruction, Hologram(Ctrl+O) kısımları gözükmüyor" dedi — menü aslında oradaydı ama yanlış yerde (sistemin global menü'sünde, Python process frontmost iken). Bundled `.app` değil de interpreter-launched Python olduğu için macOS sistem menüsü generic "Python" sembolü gösteriyor, uygulamanın kendi File/Tools/Help menü'leri görünmez kalıyor.

**Root cause**: macOS'un Cocoa NSApplication menü-bar semantiği: native menü bar her zaman ekranın tepesinde, **içinde olan uygulamaya göre değişir**. Dear PyGui 2.x `viewport_menu_bar` bu native bar'a map ediyor — bundling yoksa uygulamanın Cmd+Q gibi öğeleri burada görünür ama File/Tools gibi custom menüler proje-launcher Python process'i macOS tarafından "app" olarak tanınmadığı için belirsiz hale geliyor. Windows/Linux'ta viewport menü bar pencere içinde çizildiği için sorun yok — bu yüzden cross-platform bir trap.

**Rule**: v2 gibi interpreter-launched Dear PyGui uygulamalarında menü bar'ı **window-level** kur: `dpg.window(menubar=True)` + içinde `dpg.menu_bar()`. Bu, pencerenin title-strip'inin hemen altında menü çizer — platformdan bağımsız görünür. `viewport_menu_bar()` sadece bundled `.app`/`.exe` için uygun; geliştirme / çift-tık command launcher akışında asla kullanma. Test smoke'ları bu vakayı yakalayamaz (stub her iki pattern'ı da no-op'lar); manuel GUI test şart — launcher'ı gerçek mac'te ilk açılışta menünün **pencere içinde** görünüp görünmediğini kontrol et.

---

## 2026-04-24 — Dear PyGui `file_dialog` macOS'ta sessizce başarısız — tkinter native picker kullan

**Mistake**: v2.0.1'den v2.0.7'ye kadar "Load hologram…", reference, batch directory, export dialog'ları hep `dpg.file_dialog(...)` + custom callback üzerinden çalışıyordu. Kullanıcı pilot'ta "hologramı yüklemiyor" dedi — menü item tıklanıyor, dialog açılıyor gibi görünüyor ama dosya seçimi ya hiç tetiklenmiyor ya da callback'e boş `app_data` geliyor. Stub testleri yeşil çünkü MagicMock her şeyi no-op'luyor, runtime'da sessiz fail.

**Root cause**: Dear PyGui'nin `file_dialog` widget'ı **native değil** — DPG'nin kendi çizdiği custom bir dialog. macOS Cocoa entegrasyonu eksik: (a) file list populate edilirken flickering + seçim jsonlanmıyor, (b) callback'e bazen `file_path_name` yerine `current_path` geliyor, bazen hiç gelmiyor, (c) "Ok" butonu multi-select state'de inert kalabiliyor. Kullanıcı bu davranışları "çalışmıyor" olarak yaşıyor, hata mesajı görünmüyor çünkü callback hiç çağrılmıyor.

**Rule**: Native OS picker **her zaman** tercih. Python stdlib'inde `tkinter.filedialog` var ve **macOS'ta native NSOpenPanel/NSSavePanel**, Windows'ta Win32, Linux'ta GTK çağırıyor. Kullanım pattern'ı:
```python
import tkinter
from tkinter import filedialog
root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)
try:
    path = filedialog.askopenfilename(title=..., initialdir=..., filetypes=[...])
finally:
    root.destroy()
```
`askopenfilename` / `asksaveasfilename` / `askdirectory` tüm vakaları kapsar. `root.destroy()` **mutlaka** — yoksa Tk event loop arka planda takılı kalır. DPG'nin kendi dialog'unu tamamen kaldırma; native'ın başarısız olduğu bir edge case (container/sandbox) çıkarsa fallback olsun. Callback parsing'de her `file_path_name` | `current_path` | `selections.values()[0]` üçlüsünü destekle — DPG versiyonları arasında payload shape değişiyor.

**Critical follow-up (aynı gün bulundu)**: Yukarıdaki "inline tkinter" pattern'ı **macOS'ta Dear PyGui callback thread'inden çağrılırsa SIGTRAP** atar. Crash dump: `Thread 1 crashed: dispatch_assert_queue_fail → islGetInputSourceListWithAdditions → TSMGetInputSourceProperty → TkpInitKeymapInfo → Tk_Init`. Tk init macOS HIToolbox'ı kullanıyor, HIToolbox main-thread-only API. Dear PyGui'nin render/callback dispatcher'ı ayrı bir thread, process main thread değil. Inline `tkinter.Tk()` → anlık crash.

**Corrected rule**: macOS'ta native picker çağırırken **asla inline tkinter** kullanma. İki güvenli yol:
1. **`osascript`** (önerilen) — AppleScript `choose file` / `choose folder` / `choose file name`. Kendi process'inde çalışır, Finder main thread'ini kullanır. `subprocess.run(["osascript", "-e", script], capture_output=True)`. Native, hızlı, bağımlılık sıfır. Path `POSIX path of (choose file with prompt "..." default location POSIX file "...")`.
2. **Tkinter subprocess** — cross-platform fallback. `subprocess.run([sys.executable, "-c", driver_script], ...)` + `json.dumps` stdin / stdout. Her subprocess kendi main thread'ine sahip, HIToolbox tatmin olur. Bu yaklaşım Linux/Windows'ta da çalışır (oralarda inline da çalışır ama subprocess protocol'ü kod yolunu tekdüze tutar).

Inline tkinter yalnızca **proje main thread'i** biliniyorsa güvenli (örn. tkinter GUI app'lerinin kendi içinden). Qt/wx/DPG/ImGui gibi GUI'lerin callback context'inden **asla**. Bu kuralı script-level check ile zorla: `assert threading.current_thread() is threading.main_thread()` inline tkinter'dan önce — test'lerde yakalanır.

---

## 2026-04-24 — v1→v2 port audit: `_prepare_field` `n=1.0` hardcoded, medium RI hiç propagate'e ulaşmıyordu

**Mistake**: v2.0.3'te `ReconParams.n_medium` eklendi ve sidebar'da user-editable yapıldı. Ama `ui2.workers._prepare_field` içinde `base = ReconstructionParams(... n=1.0)` hardcoded. Yani autofocus/multi-focus/depth hepsi propagate'i n=1.0 (vakum/hava) ile çalıştırıyordu — kullanıcı n_medium=1.33 yazsa bile. `ReconstructionDriver._run` aynı şekilde `n=1.0` vardı. Sulu kültür buffer'ındaki cell örnekleri için effective wavelength yanlış hesaplanıyor, focus z sistematik olarak kayıyordu.

**Root cause**: n_medium field'ı eklendiğinde sadece QPI (cell-height math) tarafında kullanıldı. Propagation kernel'i unutuldu — `ReconstructionParams.n` alanı mevcuttu ama dataclass init'inde literal `1.0` bırakılmıştı. QPI sonuçları tutarlı göründüğünden (n_medium QPI math'inde kullanılıyor), sessiz kaldı. Sentetik sphere validation testi (n_sphere=1.40, n_medium=1.33) ile ortaya çıktı: n=1.0 ile reconstruction edilince autofocus 3.7mm kayıyordu.

**Rule**: Her `ReconParams` field'ını eklerken **tüm** downstream çağrılarını grep et. Bizim durumda `grep -n "ReconstructionParams(" src/ui2/` yaparak hem workers hem reconstruction driver'larda `n=` literal'ını bulup `n_medium`'la değiştirmeliydim. Yeni bilimsel parametre eklendiğinde "UI'a ekledim ama pipeline'a geçmedi" yaygın kaçış — port audit'inin **2. ayağı**: (1) parametre UI'da mı? (2) **Tüm** çağrı sitelerine eriştiriliyor mu? Birincisi ship, ikincisi unutulabiliyor.

---

## 2026-04-24 — Adaptive autofocus algoritmaları v1→v2 portunda kaybolmuştu

**Mistake**: v1 focus_tab'da altı autofocus algoritması vardı (linear sweep / golden / coarse-to-fine / robust / adaptive_gradient / adaptive_ratio / adaptive_bracketing). v2.0.2 port'u sadece `autofocus_zscan` (linear sweep, uniform grid) kullanıyordu. `core/autofocus/search_adaptive.py` + `core/autofocus/search_classic.py` modülleri duruyordu, sadece v2'den çağrılmıyordu. Kullanıcı "Adaptive distance ve adaptive steps vardı önceki versiyonda" diye sordu, restore etmek zorunda kaldık.

**Root cause**: v2 port'unda "autofocus = tek bir zscan çağrısı" varsayımıyla başladım. v1'in çok algoritmaya sahip olduğunu incelemeden minimum viable pipeline yazdım. Grid-search genelde işe yarar ama belirli sahnelerde (sub-step accuracy gereken, geniş z range'inde narrow peak) adaptive algoritmalar 3-10× daha hızlı/doğru sonuç verir. "Eksikliği fark edilecek kadar belirgin değil" iltifatımla gizli kaldı, kullanıcı pilot testinde yakalanınca ortaya çıktı.

**Rule**: v1'in her sidebar widget'ını sadece field olarak değil, o field'ın **davranış** sonuçlarını da listele. "algorithm=Linear Sweep" bir combo-box değeri değil, **bir worker dispatcher dalı**. v1→v2 port audit'i "widget parity" yerine "behavioural parity" olarak yap: her combo seçeneğine bakıp "bu seçenek ne iş yapıyor, v2'de aynı iş yapılıyor mu?" sor. Combo ismi eşleşse bile implement edilmemiş seçenek = kaybolmuş feature. Port tamamlandığında, v1'in rare-path'lerini (not-default adaptive, not-default unwrap, vs) içeren bir regression scenario test suite yaz.

---

## 2026-04-24 — `compute_depth_map` amplitude-only kernel'leri saf-faz numune için ters çalışıyor

**Mistake**: `core/depth_map.py::_LOCAL_KERNELS` yalnızca `LAPLACIAN_VARIANCE` ve `TENENGRAD` içeriyordu; her ikisi de `|field|` (amplitude) üzerinde operate ediyor. Saf-faz transparan numune (simülasyon mikroküre, Δn ≈ 0.07) için **amplitude fokusta uniform** — Fresnel halkaları DEFOKUSTA çıkıyor. `np.argmax` argmax aldığı için `z_map` yanlış planı — focus yerine defokus — döndürüyor. Sentetik sphere validasyon testi bunu ortaya çıkardı; production cell örnekleri (Δn + gerçek absorpsiyon) tesadüfen amplitude'dan çalıştığı için önce görünmedi.

**Root cause**: Depth localisation kriteri sample-bağımlı. Bio cells (amplitude + phase karışık) için gradient/Laplacian iyi, saf faz için TAMAMEN YANLIŞ. Kernel seti restrictive kaldığı için faz-domine senaryo map olmadı.

**Rule**: Çoklu fokus-metric'i depth map için de desteklenebilir tutmak lazım — amplitude-only kernel'lerin yanına **kompleks alan-tabanlı** bir kernel ekle. Yeni `_local_phase_variance`: `Var(Re(field)) + Var(Im(field))` üzerinden sliding window; faz-only numunede doğru yönde tepki verir, amplitude-only numunede de reasonable. `FocusMetric.PHASE_VARIANCE` key'i altında register. User'a seçim ver, pure-phase senaryosunda bu metric'i öner. Validation test suite (`tests/test_focus_validation.py`) bu ayrımı pin'liyor, gelecekteki depth map değişiklikleri sentetik sphere'lerde sessizce kırmasın.

---

## 2026-04-24 — Sentetik validasyon testi fiziksel limitlerin içinde kalmalı

**Mistake**: İlk focus validation testini yazarken toleransları dar bıraktım (scan step × 1-4), farklı sample-metric kombinasyonlarını iterate etmeden geniş z aralıkları seçtim, lateral boyut ölçümünü focused amplitude üzerinde denedim (saf-faz için düz). Sonuç: 5 test fail, root cause belirsizdi — algoritma bug'ı mı, test tasarımı mı?

**Root cause**: Sentetik validation'ın amacı algoritma davranışını ground-truth'a karşı konfirm etmek, precision benchmark'ı değil. Dar tolerance'lar fizik bias'ıyla (Fresnel envelope metric peak'ini ~2-3 step kaydırır) çakışır, test flaky oluyor. Amplitude'da saf-faz cismin çapını ölçmek → 0 ölçüm (amplitude düz). Unwrap-edilmemiş fazı peak-to-peak ölçmek → reconstruction edge artifact'leri OPD'yi 14× şişirir.

**Rule**: Sentetik validasyon tasarımında:
1. **Metric seçimi sample'a uygun**: saf-faz için `ENTROPY` (minimize) + `PHASE_VARIANCE` (yeni kernel); amplitude-carrying için `LAPLACIAN_VARIANCE`.
2. **Tolerance'lar physics-aware**: autofocus/multifocus/depth için **5-10× scan step**. Lateral boyut için **200%** (ASM'nin pixel-limited lateral çözünürlüğü 1-2 pixel blur ekler). QPI OPD için **±50%** ve ROI-mask uygulanmış (kenar reconstruction artifact'leri dışarıda).
3. **Scan parametreleri ground truth'a uygun**: sphere.z ± 1 mm dar scan / 40 step sınıf-tipik bio DHM için doğru granülerlik verir. ±25 mm wide scan (tomography range'i) local minima ve gradient varyasyonuna maruz kalır.
4. **Faz büyüklüğü 2π altında**: Δn × 2r < λ/2 seç (örn. Δn=0.005, r=10µm → peak 0.63 rad), unwrap fail mode'larını testin dışında tut. QPI OPD testi için şart, lateral için tavsiye.
5. **Çoklu obje → tek obje-per-test**: Fresnel halkalarının birbirine sızması fiziksel limit; multi-object depth map v2.1 roadmap'inde. Validation tek-obje case'iyle pipeline correctness'i pin'ler.

Bu kurallar `tests/test_focus_validation.py`'de donduruldu — gelecekteki validation test'leri onu template olarak kullanmalı.

---

## 2026-04-28 — Fine-tune training data kategori-bazlı modülerlik

**Olay**: AI asistan fine-tune'u için training örneği sayısını 17 → 100 çıkarmak gerekti. İlk plan: tek bir büyük `build_examples` fonksiyonunda 100 örneği ardışık yazmak. Sonuç: tek bir `Write` çağrısıyla 1500+ satır basmak imkansız (linter race / file modification mismatch / tool size sınırları), `Edit` ile parça parça yazmak ise old/new string eşleştirmesi her seferinde fragile.

**Root cause**: Tek mega-fonksiyonun değişiklikleri atomic olmuyor. Yeni kategori eklemek = mevcut 17 örneği + sıralı 83 yenisini içeren büyük bir blok. Hata olursa hangi kısımda olduğu belirsiz, partial-write riski yüksek.

**Rule**: Çok-örnekli training/test data üreticilerinde **kategori başına ayrı fonksiyon**:

```python
def build_examples(tools):
    out = []
    out.extend(_tool_selection_examples(tools))     # 15
    out.extend(_chain_examples(tools))              # 25
    out.extend(_self_correction_examples(tools))    # 8
    # ...
    return out
```

Avantajlar:
1. Her kategori bağımsız Edit ile büyütülür/küçültülür (~100-200 satır kontrollü).
2. Kategori eksilmesi/eklenmesi build_examples'da tek satır değişimi.
3. Test'ler kategori-bazlı isolation yapabilir (`_self_correction_examples` çıktısının arg validity bilinçli düşük olduğunu eval skoru bilir).
4. Hardware-bağlı kapsam değişikliği (stage tools askıya alma) kategori-orthogonal: `_filter_hardware()` schema'yı filtreler, kategori fonksiyonları aynı kalır.

`scripts/ai_training_examples.py`'de bu pattern dondu: 9 kategori + 1 holdout = 10 fonksiyon, her biri 30-200 satır arası. `--include-stage` / `--include-devices` flag'leri tool schema'sını hardware geri açıldığında tek satırla geri açar; örnekler aynı kalır, model schema'da yeniden görünen tool'a doğal yanıt verir.

---

## 2026-04-28 — Self-correction training örnekleri eval threshold'unu düşürür

**Olay**: Holdout'ta 1 case (H9 — "depth map laplacian variance" lowercase enum) JSON-Schema validation'ında bilerek fail ediyor (model gerçek runtime'da hatayı gördükten sonra `LAPLACIAN_VARIANCE` ile düzeltiyor). Ama `score_arg_validity` her tool çağrısını sayar → 23/24 = %95.8, AI_FINETUNE_DATA.md'nin %98 threshold'u altında.

**Root cause**: Self-correction kategorisinin doğası gereği training/holdout verisi *bilinçli invalid* args içeriyor — bu öğrenme amacının kendisi. %98 args validity threshold'u "ideal model" varsayımı yapıyor ama gerçek lab kullanımında modelin hata-sonrası recovery patterns'ı görmesi gerekiyor.

**Rule**:
1. Eval threshold'larını **training data dağılımına uygun** kalibre et — abstract bir hedef değil. Veride %4 invalid-by-design call varsa, threshold ≤ %96.
2. Threshold'u koda gömerken yorumla **niye** o değer olduğunu açıkla (`tests/test_ai_finetune_eval.py:ARG_VALIDITY_THRESHOLD` üstündeki yorum block buna örnek).
3. Alternatif: scorer'ı "post-correction final call" mantığına geçir — recovered chain'lerde sadece son success-ending call'u say. Daha akıllı ama daha kompleks. Şimdilik threshold approach yeterli.

---

## 2026-04-28 — Hardware tool'larını training schema'sından çıkarma

**Olay**: Lab profili konuşması sırasında ortaya çıktı: motorize stage henüz takılmamış, lab manuel kullanıyor. AI panel kayıtlı 28 tool'un 17'si hardware-bağlı (8 stage + 9 device: shutter / LED / acquire_grid). Bu tool'lar training data'da örnekleniyorsa model çağıracak, gerçek runtime'da hardware error dönecek → model "uydurma davranış" öğrenir.

**Root cause**: LLM ne görürse onu kullanmayı öğrenir. Tool schema'sında `goto_cell` görünüyorsa, training örneği olmasa bile, prompt "hücreye git" derken o tool'u tahminle çağırır. Hardware bağlı değilken bu garbage-in-garbage-out döngüsü.

**Rule**: Hardware-bağlı tool'ları training schema'sından **opt-out flag** ile çıkar:

```python
STAGE_TOOL_NAMES = frozenset({...})       # 8 motor-bağlı
DEVICE_TOOL_NAMES = frozenset({...})      # 9 shutter/LED-bağlı

def _filter_hardware(tools_schema, include_stage, include_devices):
    excluded = set()
    if not include_stage:   excluded |= STAGE_TOOL_NAMES
    if not include_devices: excluded |= DEVICE_TOOL_NAMES
    return [t for t in tools_schema if t["function"]["name"] not in excluded]
```

Hardware geldiğinde flag açılır, schema yeniden zengin görünür, training data'ya yeni kategori örnekleri eklenir, regenerate edilir. Mevcut training fonksiyonları değişmez. Bu **kapsam değişikliğini codebase'in geri kalanından izole eder**.

Ek not: refusal kategorisinde "Stage'i 999 mm'e götür → motorize bağlı değil" örneği var. Model schema'da `stage_move_absolute` görmediği için bunu *direkt text refusal* olarak öğrenir, hardware sonra geldiğinde bu örnek replace edilmeli yoksa model hardware varken bile reddetmeye devam eder. Hardware geri açma checklist'i: `--include-stage` flag, ek kategori örnekleri, refusal örneklerini güncelle.

---

## 2026-05-05 — Reffree benchmark'ında autofocus mismatch RMSE'yi haksızca şişiriyor — z'yi sabitle

**Mistake**: İlk reffree benchmark her iki pipeline'ın (ref-based ve reffree) **kendi autofocus**'unu çalıştırıyordu. RMSE = 4.96 rad çıktı; "classical reffree yetersiz" sonucuna varıldı. Ama gerçekte 14 frame'in bir kısmında ref autofocus +12 mm seçiyor, reffree -11 mm seçiyordu — apples-to-oranges karşılaştırma. Aynı holograma farklı z'lerden bakınca diff doğal olarak büyük.

**Why**: Autofocus metrik (Tamura/sharpness) referanssız-divide olmadığında farklı surface'ye fit eder. İki pipeline'ın "best Z"si farklı çıkmasında bg-subtraction kalitesi DEĞİL, autofocus convergence farkı baskın. Karşılaştırma kalitesini ölçmek istiyorsan **z'yi sabitle**, yalnızca araştırdığın değişkeni varsiyon yap.

**Fix**: `scripts/benchmark_reffree.py`'a `--fixed-z-from-gt` flag'ı eklendi → her iki pipeline aynı z'de (GT manifest'inden okunan) reconstruction yapıyor. Düzeltilmiş RMSE: 3.92 rad (~%20 düşüş). Hâlâ hedefin 26x üstünde, dolayısıyla classical floor'un yeterli olmadığı sonucu **yine de doğru** — ama önceki sayı yanlış sebepten yüksekti.

**Bonus bulgu**: GT manifest'inin kendisinde 10/63 frame'de autofocus yanlış z'ye gitmiş (s7 frame'lerinin çoğu -11 mm, s9_B karışık). Bu frame'ler "GT" sayılamaz çünkü kullanıcının ref-based pipeline'ı da onlarda hata yapmış. `--filter-z-outliers` flag'ı session median'dan 5 mm uzakta veya zıt half-space'teki frame'leri otomatik atıyor.

**Rule**:
1. **Kontrolü tek değişkene kilitle**: A vs B karşılaştırmasında, A ve B ortak pre-processing adımlarını **paylaşmalı**, sadece test edilen modülde ayrılmalı. Autofocus karşılaştırmıyorsan, autofocus'u sabit yap.
2. **GT'yi de skeptik gözle bak**: GT bir başka algoritmanın çıktısıysa (manuel review değilse), her frame'in GT'si geçerli olmayabilir. Outlier detection (session median, half-space sign) GT'ye de uygulanmalı.
3. **Visualization > metric**: RMSE düştüğünde "iyileşti" demeyin; diff görselini de görmeden. Bu vakada görsel bize residual'in **structured stripes** olduğunu gösterdi → polynomial basis modelleyemez, küçük CNN kolayca öğrenir → Track C tercihi.

**Pin**: `tests/test_benchmark_reffree.py::test_fixed_z_mode_locks_z_for_both_pipelines` — fixed-z mode'un ref ve reffree'ye aynı z'yi geçirdiğini ve `z_disagreement_mm == 0` olduğunu assert ediyor.

---

## 2026-05-05 — Classical reffree'nin residual'i structured (random değil) — Track C burada doğru hamle

**Olay**: Yukarıdaki düzeltilmiş benchmark'tan sonra diff görselleri (`_benchmark_reffree/diagnose_fixed_z.png`) **yatay/dikey şerit deseni** gösterdi. Random speckle DEĞİL — coherent, reproducible, session-stable bir pattern. PSD log-log'da 1/f decay; polynomial order 5'in modelleyebileceği frequency band'ının çok ötesinde içerik var.

**Why**: Off-axis hologramda referans-divide aşağıdakileri tek operasyonda kaldırır:
1. Illumination beam profile (Gaussian wavefront curvature + intensity envelope)
2. Demodulation carrier residual (sideband mask kenarından sızan hf content)
3. Sensor fixed-pattern noise (pixel response inhomogeneity)
4. Tüpteki/optikteki **statik** speckle ve toz

Polynomial/Zernike global low-order surface fit (1) ve (2)'nin DC/low-freq kısmını kapatır ama (3) ve (4) (mid-freq structured content) için kapasite yetmez. Bu yüzden pür klasik approach'ta floor ~3.5-4 rad'a takılı.

**Rule**: Reference-free residual analizi yapacaksan diff'in **frequency content + spatial pattern**'ini görselleştir. Eğer:
- Random/white → physics-side (poisson noise, shot noise) — zor problem, çok veri gerekir
- Structured/striped/banded → reproducible aberration — küçük CNN çok az veriyle (5-50 örnek) öğrenir
- Smooth gradient → polynomial order'ı artır, daha fazla DL gerekmiyor olabilir

Bu ayrım **mimari seçimini direk belirler** — pure DL (Track B) random residual için, hybrid (Track C) structured residual için.

**Pin**: Bu lesson'ın kendisi pin görevi görüyor; `tasks/track_b_pure_dl_notes.md` Track B'ye gerekenleri kayıt altına aldı, gelecekte sample diversity 8 sample tipini geçtiğinde + frame sayısı 5000+'e çıktığında re-evaluate edilir.

---

## 2026-05-05 — "Reference" hologramı bakteri içeriyor → temporal median ile sentetik temiz ref ürettik

**Kullanıcı düzeltmesi**: Track C dataset'i kuruyordum, GT olarak her session'ın "ref" PNG'sini (DHM_*.png) kullanıyordum. Kullanıcı uyardı: "elindeki görsellerde bakteriler de var saf referans değil". Yani kullandığım "referans" frame'lerin kendisi de **bakteri sinyali** içeriyor; `sample / reference` bölmesi bakteriyel yapıyı sample reconstruction'una **negatif olarak baskılıyor** — GT zaten kirli.

**Kanıt**: `_benchmark_reffree/reference_inspect.png` — s1'in "ref" frame'inin amplitude rekonstrüksiyonunda bakteri-benzeri parlak/koyu noktalar bariz görünüyor. CNN kirli GT'ye eşleşmeyi öğrenirse, gerçek samples'ta yanlış desen üretir (bakteri bilgisini "background residual" olarak öğrenir).

**Çözüm**: **Temporal median** sentetik referans:
- Bir session'daki tüm hologramları stack'le, piksel-piksel median al
- Hareketli bakteriler median'da kaybolur (rastgele konumlar → median bunları yok sayar), stabil illumination + sensor pattern + statik toz kalır
- Sonuç: sahte ama temiz "reference" — hiç sample yokmuş gibi davranıyor

`scripts/build_synthetic_refs.py` her session için median hesaplıyor; `_synthetic_refs/<session>_median_ref.png` olarak uint16 PNG yazıyor. Track C dataset builder default olarak bu sentetik ref'leri kullanıyor (`--use-synthetic-refs` ON by default). Kirli ref'lerle karşılaştırma için `--no-synthetic-refs` flag'ı bırakıldı.

**Kavramsal sonuç**: live imaging'de "saf referans" hiç olmayabilir. **Hareket = istatistiksel temizleyici**. N hareketli sample'tan median almak, optik yolda hiç sample yokken çekilmiş referansla aynı bilgi içeriği taşır (illumination + sensor karakteristikleri). Bu pattern N≥4 ve sample motion'u ≥1 piksel/frame olduğunda çalışır. Statik kontaminasyon (slide üstünde takılı kalan bakteri) median'dan geçer — bu da bilinmesi gereken sınır.

**Rule**:
1. **GT'yi sorguya çek**: GT bir başka pipeline'ın çıktısıysa, o pipeline'ın varsayımlarını da skeptik gözle gör. "Reference hologram"un saflığı verilmiş bir gerçek değil — kontrol et.
2. **Hareket varsa median al**: Live imaging'de pure reference yokluğunda, motion-based statistical cleaning (median, mean of best-N, robust mean) gerçek bir referansa eşdeğer veya daha iyi olabilir.
3. **Fallback dokümante et**: Az frame'li session'lar (s2'de 2 frame) median synth'e elverişsiz; sibling session'ın median'ını fallback olarak kullan, fallback chain'i kodda **explicit** tut, aksi takdirde sessiz "no ref → no train pair" düşmesi olur.

**Pin**: `scripts/build_synthetic_refs.py` (median synth + min-frame guard); `scripts/build_track_c_dataset.py::_load_synthetic_refs` (fallback map); `_synthetic_refs/manifest.json` (per-session synthesis log).

## 2026-07-05 — id() tabanlı cache = GC sonrası sessiz zehirlenme

**Pattern:** `CachedReconstructor._get_field_spectrum` alan spektrumunu `id(field)` ile cache'liyordu.
Python id'leri GC sonrası yeniden kullanır: eski alan ölüp aynı adrese aynı-şekilli yeni array düşünce
cache ÖNCEKİ alanın spektrumunu servis etti → yanlış frame rekonstrüksiyonu (batch/timelapse
free+realloc döngüleri birebir bu profil). Kural: **Python nesne kimliğini cache anahtarı yapma;
weakref tut** — ölü weakref asla yeni nesneyle eşleşmez. Yakalanma şekli de ders: bug'ı üretim değil,
aynı proseste art arda koşan testler (textbook-validation → cluster_heights sıra-bağımlı kirlilik)
yakaladı. Sıra-bağımlı test hatası gördüğünde "flaky" deme; global cache/singleton state ara.
Test: tests/test_reconstruction.py::test_spectrum_cache_survives_id_reuse (B-058).

## 2026-07-05 — Qt panel: ASLA kurucuda senkron ağ/bloklayan I/O yapma

**Pattern:** ui3 AIPanel.__init__ içinde health_check() → GUI thread'inde senkron requests.get
(2s timeout). Bu (a) panel kurulumunu bloke ediyor, (b) offscreen testlerde tekrarlı kurulum
urllib3 socket teardown'ıyla yarışıp **hard segfault** veriyordu. Kural: bir widget'ın __init__'i
ağ/dosya/bloklayan I/O yapmamalı — durum göstergesi nötr başlar, prob kullanıcı eylemiyle
(veya bir background QThread ile) çalışır. Test tarafında canlı servise dokunma; pure-Python stub kullan.

## 2026-07-05 — Qt closeEvent dock içeriğine cascade ETMEZ

**Pattern:** QMainWindow kapanınca, dock'a gömülü içerik widget'larının closeEvent'i çağrılMAZ
(Qt close-cascade yalnız pencerenin kendisine ulaşır). Sonuç: panel-sahibi thread'ler/timer'lar
(kamera AcquisitionThread, AIWorker, timelapse worker, device QTimer) pencere kapanınca öksüz kalır
→ zombie thread / yarım TIFF / "QThread destroyed while running". Kural: MainWindow.closeEvent'te
her panel için explicit teardown yap (shutdown() kancası + panel.close()); dock'un closeEvent'ine güvenme.

## 2026-07-05 — Bozuk venv (yeniden-adlandırılmış proje yolu)

venv/pyvenv.cfg eski "Windsurf Projects" yoluna işaret ediyor (proje "Claude Projects"e taşınmış);
tek başına ilk-Qt-import bir pytest sürecinde bazen segfault verebiliyor (bayat derlenmiş yol).
Tam süit / Qt-free-test-önce sırası bunu atlatıyor. Kalıcı çözüm: venv'i yeniden kur (ayrı iş).

## 2026-07-06 — ui3/Qt macOS: QT_WIDGETS_RHI=0 ŞART (offscreen'de görünmez segfault)

**Pattern:** ui3'ü gerçek ekranda açınca hata/blank ekran — kök neden: app.py, v1 main.py'deki
macOS Sonoma+ workaround'ını (os.environ["QT_WIDGETS_RHI"]="0", QApplication'DAN ÖNCE) taşımamıştı.
Onsuz PySide6 6.8+ bazı widget'lara Cocoa backing-surface hazır olmadan ilk paint gönderiyor →
"QPainter::begin: Paint device returned engine == 0" + **sert segfault (Python crash dump YOK)**.
QT_QPA_PLATFORM=offscreen bu Cocoa yoluna hiç girmediği için 130+ test yeşilken sorun gizli kaldı.
DERS: yeni bir Qt entry-point'i yazarken v1'in QApplication-öncesi attribute/env ayarlarını (RHI,
AA_ShareOpenGLContexts) MUTLAKA taşı; "offscreen testler geçiyor" gerçek-ekran render'ını KANITLAMAZ.

## 2026-07-06 — Qt shell+panel çoklu-sahiplik: paylaşılan sinyal bug sınıfı

**Pattern:** ui3'te bir "shell" (MainWindow) VE zengin paneller aynı bridge sinyallerine
bağımsız bağlanınca bir bug ailesi doğdu (2. review turu, B-073/B-076/B-081/B-082):

1. **Çift-wiring (B-082):** shell + panel ikisi de `autofocus_done`/`depth_done`'a bağlı →
   op başına çift-compute (shell auto-reconstruct) ve çift-repaint (aynı viewport iki kez).
   KURAL: her sonuç sinyalinin TEK sahibi olsun. Shell yalnız kimsenin sahiplenmediğini
   (recon viewport paint) + hata/status + `get_field` için cache tutsun; domain sonucunu
   (autofocus z, depth, qpi) ilgili panel sahiplensin.

2. **Argümansız `signal.disconnect()` (B-073, KRİTİK):** geçici capturing slot'u sökmek için
   `sig.disconnect()` çağırmak o sinyaldeki *TÜM* slot'ları koparır — başka dinleyicileri de.
   KURAL: `connect()`'in döndürdüğü `QMetaObject.Connection` handle'ını sakla, `disconnect(handle)`
   ile YALNIZ kendininkini kopar. Argümansız disconnect asla kullanma.

3. **Paylaşılan busy-boolean (B-076):** iki bağımsız executor tek boolean'ı paylaşınca ilk
   biten göstergeyi erken temizler. KURAL: eşzamanlı op sayısını referans-say (lock'la —
   callback worker thread'de gelir), idle'ı yalnız sıfırda sinyalle.

4. **Öksüz QThread (B-081):** `self._thread = new` ile çalışan bir QThread'in tek referansını
   ezmek → parent widget yıkımında "QThread destroyed while running" çökmesi. KURAL: bir
   probe/worker thread'i asla öksüz bırakma; hepsini bir set'te izle, `finished`'ta temizle,
   `shutdown()` hepsini `wait()`'le. Superseded olanı `disconnect` et ama çalışmaya bırak.

DERS (genel): İki katman (shell + panel) aynı event kaynağını paylaşıyorsa, "kim neyi
sahipleniyor" kontratını AÇIK yaz. "İkisi de bağlanır, idempotent olur" varsayımı çift-iş,
sinyal-sızıntısı ve thread-lifecycle çökmesi üretir. Ayrıca: MCP/headless ikincil kanalı
GUI ile aynı davranış paritesinde tutulmalı (B-078/079/080 — cache invalidation, referans
yükleme, iki-knob bg hepsi GUI'de doğru ama MCP'de eksikti).

## 2026-07-06 — Taşınmış venv + bayat __pycache__: hayalet-yol test hataları

**Pattern:** Tam süitte 3 `test_crash_handler` hatası; traceback'ler VAR OLMAYAN
`~/Documents/Windsurf Projects/.../Hybrid/tests/...` yolunu gösteriyordu. İki katmanlı kök neden:
1. `venv/pyvenv.cfg`'nin `command =` satırı venv'in aslında `Windsurf Projects/.../Phyton/venv`
   olarak YARATILIP buraya kopyalandığını kanıtladı — venv'ler taşınabilir değildir (shebang'ler,
   pyvenv.cfg, derlenmiş yollar bayat kalır).
2. Proje `__pycache__`/pytest-rewrite .pyc'leri eski konumda derlenmişti; kaynak mtime/size
   değişmediği için pytest bayat .pyc'yi taze sandı → co_filename'i eski yolu gösteren ESKİ
   kod çalıştı (dizin silinmiş olsa bile — co_filename sadece derleme anı damgası).
FIX: `pip freeze` yedeği → venv'i AYNI yorumlayıcıyla sıfırdan kur → aynı pinli seti yükle →
tüm proje `__pycache__` + `.pytest_cache` temizle. DERS: (a) venv asla kopyalanmaz/taşınmaz,
yeniden kurulur; (b) traceback'te repo dışı/var olmayan yol görürsen önce .pyc bayatlığından
şüphelen (`pyvenv.cfg` command satırı + `grep -rl <eski-yol> __pycache__` ile kanıtla);
(c) "testler geçiyor/kalıyor" hükmü vermeden önce hangi DOSYANIN çalıştığını doğrula.

## 2026-07-06 — Venv rebuild'in ortaya çıkardığı iki gizli test-harness sorunu

**B-084 (pytest-qt zincir etkileşimi):** crash_handler DOĞRU şekilde önceki excepthook'a
zincirler. pytest-qt ortama girince (ui3/PySide6 ile) "önceki hook" test sırasında pytest-qt'nin
exception-capture'ı oldu → zincirlenen çağrı "Exceptions caught in Qt event loop" olarak
raporlanıp 3 testi assert'ler GEÇERKEN FAIL'e çekti. DERS: global hook kuran kodun testi,
runner'ın ambient hook'una zincirlemesin — kurulumdan önce zararsız no-op önceki-hook sabitle
(zincirleme davranışı zaten kendi hook'unu koyan ayrı testte kalır). Yeni bir test bağımlılığı
(pytest-qt gibi AUTO-AKTİF plugin) eklemek, ona hiç dokunmayan eski testleri kırabilir.

**B-085 (koleksiyon-sırası şansı):** `from fixtures...` yalnız alfabetik olarak önce toplanan
test_calibration `sys.path`'e tests/ eklediği İÇİN çalışıyordu; ondan önce toplanan
test_autofocus_speed_baseline hiç toplanamıyor, 5 dosya standalone çalışmıyordu. DERS: test
modülü kendi import önkoşullarını KENDİSİ kurmalı (kendine-yeterlik); "tam süitte geçiyor"
sıra-bağımlı import maskesi olabilir — arada bir dosyaları standalone koştur.

## 2026-07-06 — Kendi fix'lerini de adversarial review'a sok: 3 meta-kalıp

Bugünkü düzeltme diff'ine ikinci bir bağımsız review turu (4 boyut × 2 refuter) 9 gerçek
bug daha çıkardı — ve en öğreticileri BENİM aynı gün yaptığım fix'lerin kendi boşluklarıydı:
1. **"Fix'in simetrisini tamamla":** B-078 cache invalidation'ı set_recon_param'a kondu ama
   AYNI cache'i bayatlatan diğer iki mutator (set_reference_mode, autofocus'un z yazımı)
   atlandı. Bir invariant'ı bir mutator'da düzeltiyorsan TÜM mutator'ları tara; helper'a çek.
2. **"Thread'li fix'in kendi yarışını ara":** B-076 refcount doğruydu ama busy=False'u worker
   thread'den emit etmek kuyruklanmış bayat idle'ın taze busy'yi ezmesine izin verdi. Kural:
   paylaşılan UI durumunun KARARI teslim anında, UI thread'inde, güncel sayaçla verilmeli.
3. **"Sahiplik refactor'u global yan kanalları da tarasın":** B-083 panellere explicit toast
   verdi ama shell'in set_status auto-toast'u durdu → çift toast. Bir kanalı (statü) yeniden
   dağıtırken ona bağlı ikincil kanalı (toast) da kontratın parçası yap.
Ayrıca: 2/2 refuter'dan geçemeyen "uncertain" bulguların üçü de elle bakınca GERÇEK çıktı —
uncertain'ı çöpe atma, elle doğrula.

## 2026-07-06 — Modül taşırken test-patch'lerini yaşatma: sys.modules aliasing shim'i

**Pattern:** ui2→core/drivers taşımasında düz re-export shim'i (`from new import *`) düzinelerce
`patch("ui2.workers.X")` testini sessizce no-op yapardı: patch shim modülünün attribute'unu
değiştirir ama taşınan kod KENDİ modül global'lerinden okur. FIX: eski dosya kendini hedefe
alias'lar — `import core.drivers.workers as _t; sys.modules[__name__] = _t`. Import makinesi
exec sonrası sys.modules'ü yeniden okuyup parent-paket attribute'unu alias'a bağlar → eski ve
yeni yol AYNI modül nesnesi; patch/private-isim/kimlik hepsi korunur. Modül adlarını koruyarak
taşı (`.reconstruction`/`.workers` göreli importları değişmeden çalışır). Pin'le:
tests/test_driver_relocation.py (patch-through + kimlik + "ui2 importu geri sızmasın" testi).

**B-096 eki (aynı gün):** git mv ile içerik değişmeden taşımak bile davranış bozabilir —
`Path(__file__).parents[N]` gibi konum-bağıl mantık taşımayla sessizce kayar. Taşıma
checklist'ine ekle: taşınan dosyalarda `__file__`, `parents[`, `__name__`, logger adı,
paket-göreli kaynak yolu grep'i + derinlik değiştiyse indeksleri güncelle + yolu pinleyen test.

## 2026-07-08 — "Temiz üst-küme" hissine güvenme; denetle (versiyon audit'i gerçek bug buldu)

Phyton'u incelemeden önce erken hükmüm "Hybrid temiz üst-küme, kurtarılacak bir şey yok"tu
(fft_backend'in gerçekten üst-küme olduğunu bir spot-check ile görüp). Yine de 6-ajanlı
karşılaştırma workflow'unu koşturdum — ve **gerçek bir do-now bug çıkardı (B-098)**: Hybrid'in
çıplak `autofocus_zscan`'i düz/degenerate faz-manzarasında sessizce argmax-of-noise döndürüyordu.
Flatness mantığı Hybrid'de VARDI ama yalnız bir arama algoritmasında; ata (Phyton) onu her yola
uyguluyordu. DERS: "ata versiyon ilkel, modern versiyon her şeyi kapsar" sezgisi bir kör nokta —
eski/basit kod bazen modern kodun bir yolda unuttuğu bir teşhis/guard'ı taşır. Karşılaştırmayı
dosya:satır düzeyinde YAP, sezgiyle atlama. + Salvage'ı skeptik ele: 10 ham adaydan 9'u ya zaten
karşılanıyordu ya marjinaldi; değerli olan tek/az şeyi bulmak için hepsini elemekten geçtim.

## 2026-07-08 — "Bir tanı yalnız kullanıcının koştuğu path'te işe yarar" (B-100)

**Pattern:** Version-audit "HIGH confidence" bir salvage önerdi: Phyton'un flat-curve autofocus
tanısını Hybrid'e ekle. Körü körüne uygulamak yerine ÖNCE doğruladım (CLAUDE.md: verify, don't
trust) → tanı Hybrid'de ZATEN vardı (`focus_landscape_warning`, B-098). Ama daha derin kazınca
GERÇEK bug çıktı: driver `getattr(core_result,'warning',None)` ile yalnız artık-DEFAULT-OLMAYAN
linear zscan'e bağlamıştı; settled default 'robust' hep None dönüyordu. Yani tanı vardı ama
kullanıcının fiilen koştuğu path'te ÖLÜYDÜ.
DERSLER:
1. Audit/subagent bulgusunu ("zaten var" VEYA "hiç yok") koda bakmadan kabul etme — ajan
   `autofocus_zscan`'in gövdesine bakıp satır 98'deki `warning=...`'i, ben de driver'ın onu
   nasıl forward ettiğini kaçırabilirdim. İki katman da okundu.
2. Bir güvenlik/tanı özelliği eklerken/portlarken: "bu, DEFAULT path'te aktif mi?" diye sor.
   Default değişmişse (B-095 zscan→robust) eski path'e bağlı guard sessizce devre dışı kalır.
3. Körü körüne port = ya mevcut mantığı çoğaltma ya da (beter) yumuşak-uyarıyı sert-crash'e
   çevirip regresyon. Doğrulama hem false-positive salvage'ı hem asıl boşluğu yakaladı.

## 2026-07-08 — Review workflow'u find→verify→SENTEZ olmalı (kullanıcı düzeltmesi)

**Düzeltme:** Çok-ajanlı review workflow'unu find→verify kurdum; `{confirmed, uncertain}` ham
listesi döndü. Kullanıcı: "find>verify>sentez yapman daha iyi olurdu."
**Kural:** Her adversarial review/audit workflow'una üçüncü bir **Synthesize** aşaması ekle —
verify'dan geçen bulguları alan tek bir sentez ajanı: (1) finder'lar arası AYNI kök-sorunu
dedup/merge et, (2) CLAUDE.md ilkesine + kesişen temaya göre grupla, (3) önceliklendirilmiş
fix planı üret (şimdi-düzelt = küçük+güvenli / flag'le = büyük+riskli, gerekçeli), (4) sistemik
deseni not et (ör. tekrarlayan bir silent-degrade şekli). Ham verified-liste bir RAPOR değil;
sentez, dedup + öncelik + kesişen-tema işini ana döngüye bırakmak yerine workflow'un içinde yapar.
Not: sentez ana döngüde de yapılabilir ama workflow'un parçası olması tekrarlanabilir + ucuz
(find+verify cache'li resume ile yalnız sentez canlı koşar).

## 2026-07-08 — "flag_for_decision"ı düzeltmeden önce DOĞRULA (blast-radius + gerçeklik)

3 gerçek-karar maddesini "devam" ile ele alırken, her birinde önce doğrulama fixin şeklini/riskini değiştirdi:
- **B-111 (auto_select_metric):** "tüm autofocus seçimini etkiler, B-095 re-validasyonu ister" diye
  flag'lemiştim. Çağıranları grep'leyince yalnız v1-GUI opt-in "auto-select" toggle'ında olduğu,
  B-095 default path'inin bundan HİÇ geçmediği çıktı → re-validasyon endişesi asılsızdı, tek-satır
  fix güvenliydi. DERS: "bu değişiklik X'i etkiler" demeden önce çağıran-grafını çıkar.
- **B-112 (observe cell_count):** "en zayıf bulgu, 1/2 çürütüldü" idi. `segment_cell_phase`'i
  okuyunca GERÇEKTEN binary+keep-largest olduğu, yani bulgunun DOĞRU olduğu çıktı (çürütücü yanılmış).
  Ayrıca test_ai_tools_advanced'in cell_count==2'si FARKLI bir cell_count'muş (map_sample_grid) —
  körü körüne "test var, kırma" deseydim yanlış korurdum. DERS: "uncertain/çürütüldü" etiketi bulguyu
  çöpe attırmaz; kodu oku. Ve isim-çakışması (aynı anahtar iki farklı yerde) yanıltır.
- **B-110:** flag'de "3 çağrı yeri" demiştim; gerçekte helper 2 + `_prepare_field` 4 + depth ayrı =
  daha geniş. Yayılımı düzeltmeden ölç. Genel meta-ders: flag notundaki tahminleri de doğrula.

## 2026-07-10 — "Temiz offscreen ≠ temiz gerçek ekran": state-restore + tema bug'ları

**Pattern:** Kullanıcı "arayüzde çok hata" dedi. Offscreen widget-introspection + QWidget.grab() ile
KAPSAMLI denetledim → "base layout temiz, tek desync buldum" dedim. YANILDIM. Ekran erişimi alıp
kullanıcının GERÇEK oturumunu screenshot'layınca asıl bug'lar çıktı:
1. **Floating/overlapping docks** — `restoreState()` kullanıcının kayıtlı window_state'ini koşulsuz
   geri yüklüyordu; kayıt ESKİ bir dock şemasındandı (paneller değişti/scroll-wrap/ui2 emekli), map
   olmayınca docks ayrı yüzen pencereler olarak merkezi grid'i kaplıyordu. Fix: `_LAYOUT_VERSION` damgası
   (B-115) — şema eşleşmezse layout restore edilmez.
2. **Sarı/garish tema** — kayıtlı tema high_contrast'tı (sarı accent). Offscreen'de "dark" test ettiğim
   için görmedim.
DERSLER:
- **Offscreen test, RESTORE edilmiş kullanıcı state'ini ve NATIVE render'ı KANITLAMAZ.** Fresh-state
  offscreen temizdi; bug kullanıcının kayıtlı state'indeydi. State-bağımlı/tema-bağımlı UI bug'ları için
  kullanıcının gerçek oturumunu (kendi state dosyasıyla, gerçek ekranda) görmek şart.
- **Kullanıcı "çok hata" derken abartmıyordu; ben yetersiz kanıtla "temiz" dedim.** Kullanıcı ekran
  erişiminde ısrar etseydi diye düşünme — ben istemeliydim. Belirsizlikte gerçek-ekran > offscreen-tahmin.
- **QMainWindow saveState/restoreState UI değişince sessizce bozulur** — her zaman şema-versiyonuyla gate'le.
- Bir bug sınıfı offscreen'de görünmüyorsa (native render, HiDPI, tema, restore) → o kanalı gözlemle,
  "göremiyorum, yoktur" deme.

## 2026-07-10 (düzeltme) — Kullanıcının AYARINI "bug" sanıp izinsiz değiştirme

**Düzeltme:** Gerçek-ekranda sarı temayı görünce "tema bug'ı" sanıp izinsiz dark'a çevirdim.
Kullanıcı: "renk sarı ben yaptım kendim deniyordum" — yani high_contrast'ı kendisi seçmişti,
bug değildi. Geri high_contrast'a aldım.
DERSLER:
- **Varsayılan-olmayan bir ayar (tema, dil, boyut) ≠ bug.** Onu "düzeltmeden" önce kullanıcının
  KASITLI seçimi olabileceğini düşün. Kullanıcı state'i (tercihler) kutsaldır — ellemeden önce sor
  veya en azından haber ver, geri-alınabilir yap. Ben yedek aldım ama yine de izinsiz değiştirdim.
- **Floating dock da kısmen kullanıcı eylemiydi** ("inceliyordum") — asıl bug tema/float değil,
  float edileni GERİ KOYAMAMAKtı (keşfedilebilir put-back yok). Doğru fix: davranışı geri almak
  değil, eksik affordance'ı eklemek ("Reset panel layout", B-116).
- Genel: bir UI durumu "yanlış görünüyor" diye onu üreten kullanıcı-eylemini geri alma; kullanıcının
  o duruma nasıl/neden geldiğini anla, sonra doğru düzeltmeyi (çoğu zaman bir çıkış-yolu/affordance) ekle.

---

## Qt QSS sub-control okları: border-üçgen hilesi ve data: URI ÇALIŞMAZ (2026-07-10, B-118)

Spinbox/combobox oklarını web CSS'teki `width:0;height:0 + border` üçgen hilesiyle stillemeye
çalıştım — Qt bunu üçgen olarak render ETMEZ, native indicator bloğunu çizer (boş açık-gri kare).
Sonra `image:url(data:image/png;base64,...)` denemek istedim; onu da yazmadan önce AMPİRİK test ettim.
İzole bir 4-yaklaşım harness'i (her spinbox farklı QSS) grab'leyip baktım:
- border-üçgen → kare (kötü)
- `image:url(data:...)` → HİÇBİR ŞEY (Qt QSS loader URL'yi dosya yolu sanıyor, data URI'yi çözemiyor)
- `image:url(/gerçek/dosya.png)` → net üçgen (doğru)
- ok gizle (`width:0`) → düz alan

DERSLER:
- **Qt stylesheet ≠ web CSS.** `::up-arrow`/`::down-arrow`/`::down-arrow` için tek güvenilir yol
  gerçek bir görüntü DOSYASI (`image:url(path)`). border-üçgen ve data-URI Qt'de sessizce başarısız.
  Hafızadan "muhtemelen çalışır" diye yazma — 3 satırlık bir harness'te çalıştır-gör (bu CLAUDE.md
  "recognition is not knowledge" kuralının tam örneği; işe yaradı).
- **HiDPI:** ikonu logical boyutun 2x'inde çiz, QSS `width/height`'ı logical bırak → Retina'da net.
- **Saf fonksiyonu kirletme:** `build_qss` testte QApplication'sız çağrılıyor; QPixmap QApplication
  ister. `QApplication.instance() is None` guard'ıyla headless'ta None döndür + fallback — testi
  bozmadan gerçek app'te ikon üret.
- **Kullanıcının canlı app'ı disk-state'inin ilerisinde olabilir** (in-memory workflow_mode ≠ kayıtlı).
  Salt bir stil değişikliğini "göstermek" için onu izinsiz restart etme — görünmeyen in-memory
  durumunu kaybettirirsin. Değişiklik bir sonraki açılışta gelir; öyle raporla.

---

## Çok-ajanlı görsel denetim + adversaryel doğrulama gerçek defect buluyor (2026-07-10)

Arayüzü "toparla" derken tek bir arrow-fix'le yetinmeyip TÜM yüzeyleri render edip 10 paralel
ajanla (görsel + kod) taradım, her bulguyu adversaryel doğrulamadan geçirdim (default-reject).
56 ham bulgu → 38 doğrulanmış; 18'i elendi. Bulgular gerçekti (QPI tablo kırpması bağımsız olarak
benim de gördüğümü doğruladı) VE bir toast renk BUG'ı ortaya çıkardı.

DERSLER:
- **"Toparla/temizle" gibi geniş istekte tek yüzeye bakma.** Her modu/dock'u/dialog'u/temayı
  render edip sistematik tara — kapatınca arkadan çıkanları ancak böyle görürsün (kullanıcının
  "kapattıkça diğerleri çıkıyo" şikayeti tam buydu).
- **Tek ajan bulgusuna güvenme — doğrula.** Bir MEDIUM bulgu (#2 "recon okları kopuk") YANLIŞ
  POZİTİFTİ; kodu + pikselleri kendim inceleyince recon'un qpi ile aynı olduğunu gördüm. Ajan
  bir divider çizgisini "boşluk" sanmış. Source hierarchy: run&observe > read > ajan-raporu.
- **Doğrulayıcıya kullanıcı-tercihini koru talimatı ver.** Verify prompt'una "sarı high_contrast
  teması kullanıcının kasıtlı seçimi, asla defect sayma" guard'ını koydum (önceki ders [[tema-bug-değil]]).
- **Toast LEVEL sözlüğü ≠ ROLE sözlüğü.** `toast(msg,"danger")` sessizce mavi 'accent'e düştü —
  "danger" bir ROL (buton/label rengi), toast LEVEL'ı değil (info/ok/warn/error). Yanlış level
  crash etmez, yanlış renk verir. Enum/sözlük eşleşmelerinde `.get(x, fallback)` sessiz-degrade
  tuzağıdır — geçerli anahtar kümesini test et.
- **Checkable QGroupBox iki şey ister:** ::indicator'ı AÇIKÇA stille (yoksa native indicator koyu
  temada kaybolur) + collapsed-state kuralı (yoksa gizli içeriğin etrafında boş çerçeve kalır).
- **Qt sub-control geometrisini görselden değil koddan doğrula.** İki panelin "farklı göründüğü"
  iddiasında, ikisinin de düz QDoubleSpinBox + aynı global QSS olduğunu koddan teyit ettim → aynılar.
- **Çok-ajanlı iş bittiğinde journal.jsonl'i oku**, cached/boş sonuç varsayma (workflow sonucu 157KB,
  truncate oldu — tam listeyi dosyadan parse ettim).

---

## Pencere ekrandan uzun açılınca alt butonlar erişilmez oluyor (2026-07-10, B-124)

Kullanıcı gerçek-ekran screenshot'ı attı: sol kontrol paneli ekranın altını taşıyor, en önemli
butonlar (Reconstruct/Autofocus) ekran altında, scrollbar bile yok. İki birleşik neden:
1. **Uzun paneli doğrudan dock'a koymak** (`dock.setWidget(panel)`) → panelin minimum yüksekliği
   pencerenin minimum yüksekliği oluyor → pencere paneldan kısa olamıyor → kısa ekrana hiç sığmıyor.
   Çözüm: dock'u QScrollArea'ya sar (feature dock'lar gibi) → taşan kısım kayar.
2. **`restoreGeometry` ekran-fit kontrolü yapmıyordu** → büyük ekranda kaydedilmiş geometri küçük
   ekranda ekran-dışı açılıyor. Çözüm: `_clamp_to_screen()` (restore sonrası + showEvent'te) —
   pencereyi `QScreen.availableGeometry`'ye küçültüp ekran içine çeker.

DERSLER:
- **Uzun bir widget'ı doğrudan dock/pencereye koyma; scroll'a sar.** Yoksa minimumSizeHint pencereyi
  o boyutun altına inemez yapar — kullanıcı küçültemez, kısa ekrana sığmaz. (B-113 feature panelleri
  için yapmıştı ama kontrol dock'u atlanmıştı.)
- **restoreGeometry'yi ASLA ham güvenme.** Farklı/daha büyük ekranda kaydedilmiş olabilir; her zaman
  mevcut ekranın availableGeometry'sine clamp et. Frame (title bar) margin'i show'dan sonra bilinir,
  o yüzden showEvent'te bir kez daha clamp et.
- **"Sığmıyor" şikayeti = layout değil sizing/scroll sorunu olabilir.** Panel içeriğini kısaltmaya
  çalışma; pencereyi ekrana sığdır + içeriği kaydırılabilir yap.

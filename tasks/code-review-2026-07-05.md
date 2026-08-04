# Code Review — uncommitted diff (2026-07-05)

Kapsam: `git diff HEAD` (19 modifiye) + ~30 untracked yeni dosya (Track C reffree, batch renderer, scalebar, background_phase, device_panel, depth_surface_viewer). Yöntem: 8 finder açısı × 6 aday → 1-oy adversarial doğrulama (recall-biased, high effort). Her bulgu dosya:satır doğrulandı.

> [!done] GÜNCELLEME (aynı gün): 1–7 + 10 DÜZELTİLDİ, regresyon testli, bug registry'ye işlendi (B-055…B-063)
> Süit: 1171 PASS + 10 skip (crash_handler'daki 3 hata ön-var-olan ortam sorunu — stash'li ağaçta da aynı). Registry sweep: 63 entry, FAIL=0.
> **BONUS — düzeltme sırasında yeni gerçek motor bug'ı bulundu ve düzeltildi (B-058):** `CachedReconstructor._get_field_spectrum` spektrumu çıplak `id(field)` ile cache'liyordu; GC sonrası id yeniden kullanımı aynı-şekilli YENİ alana ÖNCEKİ alanın spektrumunu servis ediyordu (batch/timelapse free+realloc döngülerinde yanlış frame rekonstrüksiyonu). Textbook-validation süiti sıra-bağımlı test kirliliği olarak yakaladı; fix: weakref-kimlik. Test: `tests/test_reconstruction.py::test_spectrum_cache_survives_id_reuse`.
> Açık kalanlar: #8 (depth ROI sample-maskeli/ref-maskesiz asimetri — tasarım kararı gerektirir) ve #9 (shape guard recon'dan sıkı — guard muhtemelen doğru taraf) bilinçli dokunulmadı; yapısal tema (reffree'yi core/pipelines'a çıkarma) ayrı bir refactor sprint'i.

## CONFIRMED — correctness/behavioral (öncelik sırası)

### 1. [HIGH] batch_renderer.py — `with_suffix` z değerini kesiyor → sweep çıktıları üst üste yazılıyor
`src/core/batch_renderer.py:339` (`_do_save`). out_base ondalıklı z içeriyor (`f"{base}_Z_{z_val:.4f}mm"` → `sample_Z_0.5000mm`, satır 455/577/507). `_do_save` bunu `base_path.with_suffix(".amp.tiff")` ile kaydediyor — `with_suffix` son noktadan sonrasını (`.5000mm`) uzantı sanıp atıyor → `sample_Z_0.amp.tiff`. **Z-sweep'te aynı tam-sayı kısmına sahip tüm dilimler tek dosyaya çök/üst üste yazılıyor; tüm sweep sessizce yok oluyor.** Fix: `with_suffix` yerine string ekleme (`base.parent / (base.name + ".amp.tiff")`) veya z'yi noktasız formatla (`z*10000:05.0f` → `um` etiketi).

### 2. [HIGH] ui2/image_panel.py — scalebar yanlış fiziksel uzunluk (bilimsel ölçüm hatası)
`src/ui2/image_panel.py:178`. `set_scalebar` `compute_scalebar(self.size, eff_pixel_um, ...)` çağırıyor ama `self.size` display texture boyutu (default 512), kaynak hologram 512'ye downsample ediliyor; plot koordinatı 0..512. `eff_pixel_um` ise **kaynak** piksel boyutu (camera/M). Sonuç: bar uzunluğu/etiketi `hologram_width/512` çarpanı kadar yanlış (ör. 2048-px hologramda 4×). **Rapora giren her ölçüm bu oranla hatalı.** Fix: ya kaynak genişliği geç (`compute_scalebar(src_width, eff_pixel_um)`) ya da pixel'i display çözünürlüğüne ölçekle (`eff_pixel_um * src_width/self.size`).

### 3. [HIGH] main_window.py — depth map "reference subtraction" checkbox'ını yok sayıyor
`src/gui/main_window.py:3927, 4118, 4269` (tomography bundle / depth overlay / depth map) `ref_field=getattr(self,'_reference_fc',None)` koşulsuz geçiyor. Reconstruct yolu (1373) `ptab.ref_enable_cb.isChecked()` ile gate ediyor; depth yolları etmiyor. Checkbox'ın toggle handler'ı yok (`_reference_fc`'i null'lamıyor). **Kullanıcı referans yükleyip (kutu auto-check) sonra "Enable reference subtraction"ı KAPATINCA reconstruct referanssız çalışıyor ama depth map hâlâ referansa bölüyor** → gösterilen rekonstrüksiyondan sessizce sapan depth. Fix: üç callsite'ı da `ref_enable_cb.isChecked()` ile gate et (reconstruct ile aynı).

### 4. [HIGH — DL track] recon_dl/losses.py — CombinedLoss device mismatch, MPS/CUDA'da her forward'da çöker
`src/recon_dl/losses.py:50-54`. Kapalı terimler `torch.tensor(0.0)` ile (CPU, `.to(device)` yok) kuruluyor ve koşulsuz `total`'a ekleniyor: `total = l1_w*l1 + char_w*char + tv_w*tv`. `char_w*char` = float×CPU-tensor = CPU; MPS/CUDA'daki `l1`'e eklenince `RuntimeError: Expected all tensors to be on the same device`. **charbonnier_weight=0.0 default olduğu için MPS/CUDA eğitimi kutudan çıktığı gibi çöküyor** — projenin Apple Silicon GPU tezini bloke ediyor (model muhtemelen CPU'da eğitilmiş). Fix: `x.new_tensor(0.0)` / `torch.zeros((), device=pred.device)` ya da `if weight>0` ile terimi koşullu ekle.

### 5. [MED-HIGH] batch_renderer.py — auto-pair "ref" içeren meşru sample'ları sessizce düşürüyor
`src/core/batch_renderer.py:136-139`. `auto_pair_reference` default True; setup `_is_reference_filename` (stem `ref_`/`ref-` ile başlar veya `_ref`/`-ref` ile biter, case-insensitive) eşleşen HER dosyayı job listesinden çıkarıyor — uyarı yok, explicit reference muafiyeti yok. **`reflow_01.tif`, `blood_ref.tif`, `ref_series01.tif` gibi meşru sample'lar sessizce hiç işlenmiyor** (ve başka bir sample'a referans olarak tüketilip onun bölmesini bozabilir). Fix: düşürülen dosyaları `status.emit` ile logla; yalnız gerçek eşleştirme yapıldığında düşür; explicit reference varken filtreyi atla.

### 6. [MED] main_window.py — autofocus/depth n_medium=1.337, reconstruct n=1.0 → z divergence
`src/gui/main_window.py:2278-2286` autofocus/depth artık `qpi_tab.n_medium` (default **1.337**) okuyor; ama `_build_recon_job` (1383) reconstruct için `n_medium=1.0` hardcode ediyor. **QPI tab'ına hiç dokunmayan kullanıcıda autofocus/depth z'si, gösterilen rekonstrüksiyona göre n=1.337 çarpanıyla kayıyor.** (Değişiklik B-022'nin v1 karşılığını kısmen düzeltmiş ama live-vs-autofocus tutarsızlığı yaratmış.) Fix: iki yol da aynı n_medium kaynağını kullansın.

### 7. [MED] ui2/app.py — DearPyGui içindeki Qt 3D surface penceresi event loop'suz (donuk)
`src/ui2/app.py:3197`. `QApplication([])` kuruluyor ama `.exec()` yok; dosyada `processEvents`/`QTimer`/`QFileDialog` yok (ui2 `dpg.file_dialog` kullanıyor, Qt event pompalamıyor). QDialog viewer yalnız `.show()`. **DPG ana loop'u sahip; Qt penceresi paint/etkileşim event'i alamaz → donuk/boş, döndürülemez.** Yorum "Qt zaten export dialog'ları için yüklü" yanlış. Fix: DPG frame loop'unda `qapp.processEvents()` pompala, ya da surface'ı matplotlib-subprocess (v1 `_show_3d_surface` deseni) ile çiz.

## PLAUSIBLE

### 8. [MED] ui2/workers.py — depth ROI: sample hard-mask'li, ref maskesiz (border tutarsızlığı)
`src/ui2/workers.py:931-939`. `run_depth_map` sample_field'i keskin 0/1 dikdörtgenle maskeliyor (apodizasyon yok), ref_field'i maskesiz geçiriyor; `propagate(masked_sample)/propagate(ref)`. Keskin maske propagasyonda ringing enjekte eder; masked-sample/unmasked-ref bölmesi ROI kenarında fiziksel tutarsız → best-z single-plane reconstruct'tan sapabilir. Not: keskin maske eski `_prepare_field`'de de vardı (ringing pre-existing); asimetri yeni. Fix: ref'i de aynı ROI ile maskele veya apodize et.

### 9. [LOW-MED] depth_map.py — katı shape guard, recon'dan daha sıkı (regresyon)
`src/core/depth_map.py:259-263` `ref_field.shape[-2:] != (h,w)` → ValueError. Recon/batch referans bölmesi `np.where` ile sessiz devam ediyor. Farklı-boyutlu referans yüklenirse depth hard-fail (main_window try/except → "Depth-map compute failed") ama recon çalışıyordu. Guard aslında daha güvenli (broadcast bozardı) ama recon'la tutarsız.

### 10. [LOW] device_panel.py — jog() 3-tuple unpack, yutulan except → sessiz no-op
`src/ui2/device_panel.py:256` `x,y,z = self.stage.position_um` (3 eksen varsayıyor), bare `try/except: return` içinde. `refresh_snapshot` (208) `len(pos)>=3` ile guard'lı (tutarsız). 2-eksenli backend → ValueError yutulur → jog sessiz no-op ama HUD güncellenir (operatör hareketti sanır). Şu an sadece mock (3-tuple) olduğu için tetiklenemiyor.

## REFUTED
- **Live vs batch subtract_mean divergence** — `_build_recon_job` anahtarı her zaman set ediyor (checkbox default True), settings_schema default True; worker'ın `False` fallback'i live yolda ölü. Sapma yok.

## Yapısal tema (cleanup/altitude — birden çok finder bağımsız işaretledi)
Reffree fizik zinciri (demodulation + per-z reference division + poly background fit) **3-4 yerde kopya**: `scripts/run_rapor_data_batch.py`, `scripts/benchmark_reffree.py`, `src/core/batch_renderer._apply_ref`, `src/core/depth_map`. Dahası **`src/recon_dl/inference.py` production kodu `scripts/`'ten import ediyor** (layering inversion) ve optik sabitler (`WAVELENGTH_M`, `DATA_ROOT=$DHM_DATA_ROOT/...`) script global'i olarak DL pipeline'ın config otoritesi olmuş.

> [!done] BÜYÜK KISMI DÜZELTİLDİ (2026-07-05) — B-064
> Reffree zinciri (`preprocess_raw`/`demodulate`/`autofocus_z`/`propagate_field`/`safe_reference_divide`/`propagate_referenced`/`reconstruct_in_memory`/`piston_align`) → **`src/core/pipelines/reffree_hybrid.py`** (parametreli `OpticalConfig`, `LAB_OPTICS` default). `inference.py` artık core'dan import ediyor (`sys.path scripts` kaldırıldı — **layering inversion kırıldı**, kaynak-seviyesi guard test'i var). `benchmark_reffree` + `run_rapor_data_batch` helper'ları core'a delege eden ince wrapper (byte-parity testli). Optik sabitler artık `OpticalConfig` — DL pipeline tek mikroskoba bağlı değil. Testler: `tests/test_reffree_pipeline.py` (7 test). **Kalan (ayrı, düşük öncelik):** `batch_renderer._apply_ref` ve `depth_map` hâlâ kendi ref-division inline'ını taşıyor (Qt/farklı imza — çekirdeğe almak ek iş); `background_phase.fit_polynomial_background` vs `phase_unwrap._remove_polynomial_bg` (iki poly-bg fit); `main_window._nice_scalebar_length` `core.scalebar`'a taşınmamış; reffree testleri hâlâ absolute Desktop yolunda (tmp_path'e çevrilmeli).

## Verimlilik (opsiyonel)
- `background_phase.py` her frame'de Zernike/polinom basis'i sıfırdan kuruyor (~230-640MB float64/frame) — (shape,n_terms) ile cache'lenebilir; 3000-frame batch'te dakikalarca boşa.
- `recon_dl/inference.py` tile'ları tek tek (batch=1) + her tile'da `.cpu()` sync — tek batched forward'a topla.
- `device_panel.py` 5Hz poll her tick yeni `threading.Timer` (yeni OS thread) + koşulsuz repaint — tek uzun-ömürlü timer + dirty-check.

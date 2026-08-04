# AI Vision + MCP + Reference-Free — Design & Plan (2026-07-05)

## Karar (kullanıcı, 2026-07-05)
- **Build sırası:** core-first (tool/gözlem katmanı önce; her iki kanala hizmet eder).
- **MCP:** full drive (read+write), güvenlik katmanı devrede, **headless** ayrı entry point (GUI'ye gömülü değil).
- **Vision:** ikisi de — numeric inspect (local LLM okur) **+** `render_view` PNG (MCP→vision model gerçek görür).
- **Reffree UI:** v2 DPG (ui2).

## Birleştirici mimari
```
core/ (Qt-free)
  pipelines/reffree_hybrid.py   ✅ (mevcut)
  observe.py                    ← Phase 1 (saf gözlem: inspect_* + render_view)
  ai/tools registry (35) + security  ✅ (mevcut)
        │                                   │
  in-app copilot (local LLM)          dhm-mcp (Phase 2, headless, full-drive)
  numeric inspect (text)              + render_view PNG → Claude görür
```
**Tek registry, iki kanal.** In-app copilot (offline, lab-içi) olduğu gibi kalır; MCP "dışarıdan Claude sürsün" kanalıdır. Kod tekrarı yok.

## Neden bu ayrım (Blender Optics dersi)
Kardeş proje `blender-optics-mcp`: MCP motoru **headless** sürer, GUI event-loop'una kuple edilmez; tool'lar yapısal veri döndürür; `inspect_beam`/`inspect_element` vision deseni. DHM karşılığı bu plan.

## Fazlar

### Phase 1 — core/observe.py (BU FAZ) + AI tool'ları
Saf, Qt-free fonksiyonlar (numpy + matplotlib-Agg). Girdi = numpy array, çıktı = yapısal dict / PNG bytes.
- `inspect_reconstruction(complex_field, ...) -> dict`: focus skoru (Laplacian var), amplitude kontrastı, faz RMS, background düzlük residual'i, finite-oranı.
- `inspect_phase_map(unwrapped, wavelength_m, pixel_m, n_sample, n_medium) -> dict`: OPD istatistikleri (min/max/mean/p95), gradient RMS, hücre segment sayısı (qpi.segment_cell_phase), toplam dry-mass.
- `inspect_field(complex_field) -> dict`: amp/faz histogram özeti, dinamik aralık, sideband gücü.
- `render_view(array, kind, pixel_um=None, ...) -> bytes`: PNG (kind = amplitude/phase/spectrum/depth); 1-99 percentile contrast (core.contrast); opsiyonel scalebar (core.scalebar). Local LLM için "kaydettim"; MCP'de image content olarak döner.

**Tool wiring (Phase 1b):** `tool_impls.py`'ye 5 tool: `inspect_reconstruction`, `inspect_phase`, `inspect_field`, `render_view`, `set_reconstruction_mode`. ToolContext'e `get_last_field()` callable eklenir (son complex field/phase array'lerini verir) — v1 panel + MCP headless context ikisi de implemente eder.

### Phase 2 — dhm-mcp (headless MCP sunucusu)
- `src/dhm_mcp/` (veya `mcp/`): `build_tool_registry()` + **headless ToolContext** (core'u doğrudan süren, Qt'siz — reffree pipeline gibi). stdio MCP.
- Full drive: load→recon→autofocus→QPI→reffree + inspect + render_view. Irreversible tool'lar confirm-gate'li (MCP'de confirm = otomatik reddet ya da client-elicitation).
- `[mcp]` extra (mcp[cli] bağımlılığı). `dhm-mcp` entry point.

### Phase 3 — ui2 reference-mode UI
- ui2 reconstruction kontrollerine 3-yönlü mod: **Off / Reference / Reference-free** (reffree → `core.pipelines.reffree_hybrid` background subtraction; saf numpy, hep var).
- DL Track C düzeltici = opsiyonel advanced toggle (torch+model varsa; yoksa gri).
- `set_reconstruction_mode` tool'u ile copilot + MCP de bu modu sürer.

## Kısıtlar / notlar
- Local model qwen2.5:7b **text-only** → görüntü göremez; gerçek "görme" MCP→vision model ile. `render_view` yine de local'de PNG üretir (operatör görür).
- ui2 AI paneli hâlâ bozuk (API imza uyumsuzluğu) — reffree UI'ı ui2'nin reconstruction kontrollerine girer, AI paneline değil; AI panel fix ayrı iş.
- Güvenlik: `render_view` çıktısı redaksiyon dışı (görüntü lab-verisi) → MCP'de opt-in/uyarı.

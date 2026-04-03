"""
QPI Integration Guide
======================

1. FILE PLACEMENT
-----------------
  core/qpi.py              → Hesaplama modülü
  gui/sidebar/qpi_tab.py   → Sidebar tab widget

2. SIDEBAR'A EKLE (sidebar_tabs.py)
------------------------------------
  from .qpi_tab import QPITab

  # __init__ içinde:
  self.qpi_tab = QPITab()
  self.tabs.addTab(self.qpi_tab, "QPI")

3. MAIN_WINDOW HOOKUP (main_window.py)
---------------------------------------
  # __init__ veya _connect_signals içinde:
  self.sidebar_tabs.qpi_tab.compute_requested.connect(self._compute_qpi)

  # Yeni method:
  def _compute_qpi(self):
      if self._recon_complex is None or self._phase_unwrapped is None:
          self.status_bar.show_message("Run reconstruction first")
          return

      from core.qpi import compute_qpi, QPIMode

      qtab = self.sidebar_tabs.qpi_tab

      mode = QPIMode(qtab.mode_combo.currentData())
      n_sample = qtab.n_sample.value()
      n_medium = qtab.n_medium.value()
      alpha_mL_g = qtab.alpha.value()
      alpha_SI = alpha_mL_g * 1e-6  # mL/g → m³/kg

      known_h = None
      if qtab.known_thickness_cb.isChecked():
          known_h = qtab.known_thickness_nm.value() * 1e-9

      rtab = self.sidebar_tabs.recon_tab
      wl = float(rtab.wavelength_nm.value()) * 1e-9
      px = float(rtab.pixel_um.value()) * 1e-6
      if not rtab.pixel_is_effective_cb.isChecked():
          mag = float(rtab.magnification.value())
          px = px / (mag if mag > 0 else 1.0)

      self.status_bar.show_message("Computing QPI...")

      result = compute_qpi(
          phase_unwrapped=self._phase_unwrapped,
          wavelength_m=wl,
          pixel_size_m=px,
          mode=mode,
          n_sample=n_sample,
          n_medium=n_medium,
          alpha=alpha_SI,
          known_thickness_m=known_h,
          cell_threshold=qtab.cell_threshold.value(),
          compute_psd=qtab.show_psd_cb.isChecked(),
      )

      # Stats panellerini doldur
      if result.phase_stats:
          qtab.set_phase_stats(result.phase_stats)
      if result.cell_morph:
          qtab.set_cell_stats(result.cell_morph)
      if result.roughness:
          qtab.set_roughness_stats(result.roughness)

      # Görüntü panellerini güncelle
      if qtab.show_opd_cb.isChecked() and result.opd_nm is not None:
          self.panel_amp.set_image(result.opd_nm, autoLevels=True)
          # veya ayrı bir QPI display paneli eklenebilir

      if qtab.show_height_cb.isChecked() and result.height_nm is not None:
          # Ayrı panel veya mevcut panellere overlay
          pass

      if qtab.show_drymass_cb.isChecked() and result.dry_mass_density_pg_um2 is not None:
          # Dry mass density haritası
          pass

      self.status_bar.show_message(
          f"QPI complete — Dry mass: {result.total_dry_mass_pg:.1f} pg"
          if result.total_dry_mass_pg else "QPI complete"
      )

4. DISPLAY PANEL ÖNERİSİ
--------------------------
  QPI modunda mevcut 4 panel şöyle değişebilir:

  ┌─────────────┬─────────────┐
  │  OPD map    │  Height /   │
  │  (nm)       │  topography │
  │  colorbar   │  (nm)       │
  ├─────────────┼─────────────┤
  │  Dry mass   │  Phase      │
  │  density    │  histogram  │
  │  (pg/μm²)  │  + stats    │
  └─────────────┴─────────────┘

  Micro-structure modunda:

  ┌─────────────┬─────────────┐
  │  OPD map    │  Height     │
  │  (nm)       │  3D surface │
  ├─────────────┼─────────────┤
  │  Roughness  │  PSD        │
  │  stats      │  log-log    │
  └─────────────┴─────────────┘

5. COLORMAP ÖNERİLERİ
-----------------------
  OPD map:       'RdBu_r' (diverging, sıfır merkezli)
  Height:        'viridis' veya 'hot'
  Dry mass:      'inferno'
  RI map:        'coolwarm'
  Phase:         'twilight' (cyclic) veya 'RdBu_r'

6. LINE PROFILE TOOL
---------------------
  pyqtgraph'ın ROI (Region of Interest) özelliğini kullan:
    roi = pg.LineSegmentROI(...)
    roi.sigRegionChanged.connect(update_profile)

  Kullanıcı çizgiyi sürükleyince otomatik profil güncellenir.
"""

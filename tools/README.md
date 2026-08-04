# DHM Reconstruction — Test Tools

Her modülü bağımsız olarak test etmek için hazır araçlar.  
Tüm tool'lar `Hybrid/` kök dizininden çalıştırılmalıdır:

```bash
cd /path/to/dhm-hybrid       # repo kök dizini
python -m tools.<tool_adı>
```

## Araçlar

| Dosya | Özellik | Açıklama |
|-------|---------|----------|
| `tool_ingestion.py` | Görüntü Yükleme | PNG/TIFF/NPY dosya okuma, metadata çıkarma |
| `tool_offaxis.py` | Off-axis Çıkarım | +1 order tespiti, spektral maskeleme, karmaşık alan çıkarımı |
| `tool_reconstruction.py` | Propagasyon | ASM ve Fresnel yöntemiyle dalga yayılımı |
| `tool_autofocus.py` | Otomatik Odaklama | Z-scan, Golden Section, Coarse-to-Fine, Robust arama |
| `tool_phase_unwrap.py` | Faz Açma | Gradient Integration, TIE, Quality Guided, Least Squares, Goldstein |
| `tool_qpi.py` | QPI Pipeline | OPD, yükseklik, kuru kütle, pürüzlülük hesaplama |
| `tool_fft_backends.py` | FFT Backend | NumPy/SciPy/PyFFTW/MLX performans karşılaştırması |
| `tool_contrast.py` | Kontrast | Percentile stretch, CLAHE, histogram eşitleme |
| `tool_exporter.py` | Dışa Aktarma | NPY/TIFF/PNG/CSV/MAT formatlarına kaydetme |
| `tool_full_pipeline.py` | Tam Pipeline | Yükleme → Off-axis → Propagasyon → Unwrap → QPI (uçtan uca) |

## Hızlı Kullanım

```bash
# Tek bir tool çalıştır
python -m tools.tool_ingestion

# Tüm tool'ları çalıştır
python -m tools.run_all

# Belirli bir sample ile çalıştır
python -m tools.tool_reconstruction --image "labtest/1-100_25um_stdSlide_withoutcovSlip_2.png"
```

## Varsayılan Test Parametreleri

- **Dalga boyu**: 632.8 nm (He-Ne lazer)
- **Piksel boyutu**: 4.4 µm (kamera) / 50x büyütme = 0.088 µm efektif
- **Yöntem**: ASM (Angular Spectrum Method)
- **Odak metriği**: phase_variance
- **Faz açma**: quality_guided

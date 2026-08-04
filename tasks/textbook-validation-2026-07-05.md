# Kitap-algoritma tutarlılık doğrulaması (2026-07-05)

Optik/DHM/holografi kitaplarından kapalı-form analitik sonuçları çıkarıp Hybrid
motoruna (`src/core`) karşı test ettik. Süit: `tests/test_textbook_validation.py`.
**Sonuç: 13 PASS + 1 SKIP (scope map). Motor tüm kapsam-içi kapalı-formlarla tutarlı.**

Koşum:
```
PYTHONPATH=src ./venv/bin/python -m pytest tests/test_textbook_validation.py -v
```

## Kaynaklar (gerçekten okunan PDF'ler)
- **Kim**, *Digital Holographic Microscopy*, Springer 2011 (251 s, temiz okundu)
- **Kreis**, *Handbook of Holographic Interferometry*, Wiley-VCH 2005 (547 s)
- **Hecht**, *Optics*, Ch.9-10 (730 s)
- **Born & Wolf** — çapraz-referans (aşağıdaki uyarıya bak)

> [!warning] İki PDF sahte/yanlış çıktı
> - `Principles_of_Optics_Electromagnetic_The.pdf` = Born & Wolf'un **3 sayfalık kitap incelemesi** (Physics Today 2000, Hecht), denklem yok.
> - `881862587-Digital-Holography-...-Full-Download.pdf` = Schnars & Jüptner **değil**; bir indirme-scam / ReportLab placeholder PDF'i.
> Bu iki kaynağın formülleri gerçek kitaplardan (Kim/Hecht/BW) çapraz-alındı.

## Yük-taşıyan formüller — physics_verify (Docker oracle) ile doğrulandı
| Formül | Oracle sonucu |
|---|---|
| OPD = φλ/2π | **VERIFIED 4/4** (DIMENSIONAL+NUMERIC) |
| h = OPD/(n_s−n_m) | **VERIFIED** (h=21.1µm @ φ=2π,dn=0.03) |
| Λ = λ/(2 sinθ), f_c=2sinθ/λ | **VERIFIED 3/3** (Λ=9.066µm @ θ=2°) |
| z_T = 2p²/λ (Talbot) | DIMENSIONAL green + numeric 202.212µm (set-wrap harness artefaktı dışında) |
| ASM eğim √(1−(λf/n)²)→1, kesim f_c=n/λ | LIMIT+DIMENSIONAL green |

## Test edilen vakalar (kapsam-içi, motora karşı)
| Test | Kitap | Motor fonksiyonu | Sonuç |
|---|---|---|---|
| OPD = φλ/2π (1 dalga → λ) | Kim/Hecht | qpi.phase_to_opd | ✅ |
| h = OPD/Δn (21.1µm) | Kim Ch.11 | qpi.opd_to_height | ✅ |
| Yansıma h = OPD/2 | Kim/Hecht | qpi.opd_to_height_reflection | ✅ |
| Forward-model round-trip | Kreis/BW | phase_to_opd∘opd_to_height | ✅ |
| Off-axis carrier +1-order @ 77 bin; Λ | Kim §7.4/Kreis | offaxis + masking.detect_plus_one_order | ✅ |
| Off-axis genlik geri kazanımı | Kim | extract_complex_field_offaxis | ✅ |
| ASM evanescent kesim \|f\|>n/λ, \|H\|≤1 | Kim §4.4/BW | reconstruction ASM H | ✅ |
| ASM düzlem-dalga fazı 2πnz/λ | BW | propagate(ASM) | ✅ |
| Propagasyon round-trip identity (ASM+Fresnel) | BW unitarity | propagate(+z)∘(−z) | ✅✅ |
| Talbot self-image z_T (Fresnel+ASM) | Kim §8.4.5/BW | propagate(z_T) corr>0.98 | ✅✅ |
| Talbot yarı-mesafe p/2 kayması | Kim/BW | propagate(z_T/2) anti-corr | ✅ |
| Faz-unwrap lineer ramp | Kreis §5.9.2 | unwrap_phase_advanced(GRADIENT_INTEGRATION) | ✅ |

**Doğrulama sırasında öğrenilen (bug değil, tasarım):** Default unwrap pipeline
`post_bg_remove=True` (order-2 düzlem fit) → saf lineer ramp'i bilerek düzleştiriyor.
Unwrapping'in kendisini (2π atlama giderme) test etmek için `post_bg_remove=False` şart.

## Kapsam DIŞI (motor bu fiziği yapmıyor — belgelendi, iddia edilmedi)
Motor **near-field angular-spectrum / convolution-Fresnel** propagatörü:
- **Fraunhofer far-field** sonuçları: Airy ilk-sıfır 1.22λf/D, tek/çok-yarık minima,
  Rayleigh/Abbe çözünürlük (0.61λ/NA) — motor far-field yoğunluk hesaplamıyor.
- **Single-FFT Fresnel-transform pixel** Δξ=λd/(NΔx) ve z_min=X0²/(Nλ) — motor
  convolution-tipi (pixel pitch'i KORUR), FTM değil. Bu formüller uygulanmıyor.

Bu kalemler `test_scope_map_out_of_scope` içinde açık `skip(reason=...)` ile kayıtlı.

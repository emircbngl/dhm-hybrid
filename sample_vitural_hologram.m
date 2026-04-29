% =========================================================================
%  OFF-AXIS FRESNEL HOLOGRAFİ — Angular Spectrum Method
%  Her piksel tüm nesne bilgisini taşır (difraksiyon ile)
%  ► 3 µm küre hologramı   (ayrı)
%  ► 25 µm küre hologramı  (ayrı)
%  Optik: M=50×, f_tube=200 mm, λ=632.8 nm
%  Saçak açısı: y-eksenine göre -45°
%  Kamera: 4.4 µm piksel, 12-bit ADC
% =========================================================================
clear all; close all; clc;
try; pkg load image; catch; end
try; graphics_toolkit('qt'); catch; end

% ─────────────────────────────────────────────────────────────────────────
%  1) ORTAK PARAMETRELER
% ─────────────────────────────────────────────────────────────────────────
lambda = 632.8e-9;           % He-Ne dalgaboyu [m]
k0     = 2*pi/lambda;
M      = 50;                 % büyütme
f_tube = 200e-3;             % tüp lens [m]
f_obj  = f_tube/M;           % 4 mm

dn     = 0.551;              % küre (n=1.551) – hava (n=1.000)

px_cam    = 4.4e-6;            % kamera piksel boyutu [m]  (varsayılan 4.4 µm)
bit_depth = 12;               % ADC bit derinliği
ADC_max   = 2^bit_depth - 1;  % 4095 say

N      = 1024;               % grid
Nz     = 100;                % hacim dilim sayısı

dx_s   = px_cam / M;         % numune pikseli = 88 nm
dx_i   = px_cam;             % image/hologram pikseli = 4.4 µm

% Hologram düzlemi: image plane'den z_h ötede.
% Fresnel difraksiyon burada devreye girer: her piksel nesnenin tamamından
% katkı alır → gerçek holografik kayıt.
z_h = 50e-3;                 % 50 mm

% ─────────────────────────────────────────────────────────────────────────
%  2) KOORDİNAT GRİDLERİ
% ─────────────────────────────────────────────────────────────────────────
x_s = (-N/2 : N/2-1) * dx_s;    % numune ekseni
x_i = (-N/2 : N/2-1) * dx_i;    % image/hologram ekseni

[Xs, Ys] = meshgrid(x_s, x_s);
[Xi, Yi] = meshgrid(x_i, x_i);

% ─────────────────────────────────────────────────────────────────────────
%  3) ANGULAR SPECTRUM TRANSFER FONKSİYONLARI
%  H_fwd : image plane → hologram düzlemi (+z_h)
%  H_bwd : hologram düzlemi → image plane (geri yayılım)
%
%  Merkezi frekans koordinatlarında tanımlanır (DC = merkez)
% ─────────────────────────────────────────────────────────────────────────
dfx       = 1/(N * dx_i);
fx_c      = (-N/2 : N/2-1) * dfx;
[FX_c, FY_c] = meshgrid(fx_c, fx_c);

kz2       = (1/lambda)^2 - FX_c.^2 - FY_c.^2;
kz2       = max(kz2, 0);         % evanescent → kesmek için sıfırla
H_fwd     = exp( 1i * 2*pi * z_h * sqrt(kz2) );
H_bwd     = conj(H_fwd);         % ters yayılım = fazı geri al

% ─────────────────────────────────────────────────────────────────────────
%  4) OFF-AXIS REFERANS DALGASI — y-eksenine göre -45°
%
%  Saçak yönü  : (1,−1)/√2  →  fringes "/" şeklinde, y'den −45°
%  Taşıyıcı yönü: (1,+1)/√2  →  fx = fy = f_carr/√2
%
%  R = exp( i·2π·( fx·x + fy·y ) )
%      saçak: x + y = sabit  →  y'ye göre −45°
% ─────────────────────────────────────────────────────────────────────────
f_carr  = 1 / (4 * dx_i);          % toplam taşıyıcı büyüklüğü (4 px saçak)
f_x_ref = f_carr / sqrt(2);         % x bileşeni
f_y_ref = f_carr / sqrt(2);         % y bileşeni  (eşit → 45°)

R_holo  = exp(1i * 2*pi * (f_x_ref*Xi + f_y_ref*Yi));

% FFT'de +1 siparişinin piksel ofseti (KX ve KY eksenleri)
k_x = round(f_x_ref * N * dx_i);   % ≈ N/(4√2) ≈ 181 px  (sütun)
k_y = round(f_y_ref * N * dx_i);   % aynı                  (satır)

% ─────────────────────────────────────────────────────────────────────────
%  5) İÇ YARDIMCI FONKSİYON: ASM ileri/geri yayılım
% ─────────────────────────────────────────────────────────────────────────
asm_prop = @(U, H) ifft2(ifftshift( fftshift(fft2(U)) .* H ));

% ─────────────────────────────────────────────────────────────────────────
%  6) NESNE TANIMLARI (iki ayrı hologram)
% ─────────────────────────────────────────────────────────────────────────
nesneler = {
  struct('r_um', 1.5,  'cap',  3, 'label', '3um')
  struct('r_um', 12.5, 'cap', 25, 'label', '25um')
};

fprintf('=== PARAMETRELER ===\n');
fprintf('λ = %.1f nm  |  M = %d×  |  f_obj = %.0f mm  |  f_tube = %.0f mm\n',...
  lambda*1e9, M, f_obj*1e3, f_tube*1e3);
fprintf('Kamera: px=%.1f µm  |  %d-bit ADC (max=%d)\n',...
  px_cam*1e6, bit_depth, ADC_max);
fprintf('dx_s = %.0f nm  |  dx_i = %.1f µm  |  N = %d  |  Nz = %d\n',...
  dx_s*1e9, dx_i*1e6, N, Nz);
fprintf('z_h = %.0f mm  |  f_carr = %.1f /mm  |  saçak −45° (y)\n\n',...
  z_h*1e3, f_carr*1e-3);

% ─────────────────────────────────────────────────────────────────────────
%  7) HER NESNE İÇİN HOLOGRAFİK SİMÜLASYON
% ─────────────────────────────────────────────────────────────────────────
for ob = 1:2
  s = nesneler{ob};
  r = s.r_um * 1e-6;

  fprintf('══════════════════════════════════════════\n');
  fprintf(' %d µm küre  |  Δφ_max = %.1f rad  (%.1f × 2π)\n',...
    s.cap, k0*dn*2*r, k0*dn*2*r/(2*pi));
  fprintf('══════════════════════════════════════════\n');

  % ── [1/4] Hacim Slicing: numune düzleminde nesne dalgası ──────────
  % Küre voxellerine k0·Δn·dz faz eklenerek alan adım adım çarpılır.
  % Bu, ince-dilim (split-step) yaklaşımıdır; volüm içi kırılmayı
  % tam anlamıyla modellemektedir.
  fprintf('  [1/4] Hacim slicing (Nz=%d, dz=%.2f nm)...\n',...
    Nz, (2*r/Nz)*1e9);

  z_arr = linspace(-r, r, Nz);
  dz    = z_arr(2) - z_arr(1);
  U_s   = ones(N, N);   % başlangıç: birim genlikli düzlem dalga

  for iz = 1:Nz
    inside = (Xs.^2 + Ys.^2 + z_arr(iz).^2) <= r.^2;
    U_s    = U_s .* exp(1i * k0 * dn * dz * inside);
  end
  % U_s: numune düzlemindeki nesne dalgası — faz nesneyi, genlik ≈ 1

  % ── [2/4] 4f Büyütme → Image Plane ───────────────────────────────
  % ─ Koordinat büyütmesi piksel adımıyla encode edildi (dx_s → dx_i)
  % ─ Faz değişmez (optik yol uzunluğuna bağlı)
  % ─ Genlik 1/M (enerji korunumu)
  O_i = U_s / M;

  % ── [3/4] Angular Spectrum İleri Yayılım ─────────────────────────
  % O_i (image plane) → O_holo (hologram düzlemi, z_h ötede)
  % Her piksel Fresnel difraksiyon aracılığıyla tüm nesne bilgisini taşır.
  fprintf('  [2/4] ASM ileri yayılım (z = %.0f mm)...\n', z_h*1e3);
  O_holo = asm_prop(O_i, H_fwd);

  % ── [3/4] Off-Axis Hologram Kaydı + Kamera Modeli ───────────────
  H_cont = abs(O_holo + R_holo).^2;
  % H = |O|² + |R|² + O·R* + O*·R  →  DC / DC / +1 / −1 siparişler

  % Kamera: 12-bit ADC kuantizasyonu
  %   Normalize [0,1] → [0, ADC_max] → tamsayı (kamera sayısı)
  H_norm  = (H_cont - min(H_cont(:))) / (max(H_cont(:)) - min(H_cont(:)));
  H_cam   = round(H_norm * ADC_max);          % uint16 benzeri [0,4095]
  H_proc  = double(H_cam) / ADC_max;          % işleme için normalize

  fprintf('  Hologram sürekli: [%.4f, %.4f]\n', min(H_cont(:)), max(H_cont(:)));
  fprintf('  Kamera çıkış    : [%d, %d] counts  (%d-bit)\n',...
    min(H_cam(:)), max(H_cam(:)), bit_depth);

  % ── [4/4] Numerik Rekonstrüksiyon ────────────────────────────────
  fprintf('  [3/4] Rekonstrüksiyon...\n');

  % i) Hologramın 2D FFT'si (merkezi koordinat)
  FH = fftshift(fft2(H_proc));

  % ii) +1 siparişini yuvarlak filtre ile seç
  %     Taşıyıcı çapraz: (+k_x, +k_y) pikselinde (sütun, satır)
  [KX, KY] = meshgrid(-N/2:N/2-1, -N/2:N/2-1);
  bw        = round(N/5);
  mask_p1   = ((KX - k_x).^2 + (KY - k_y).^2) <= bw^2;   % 2D çember filtre
  F_p1      = FH .* mask_p1;

  % iii) 2D Demodülasyon: +1 siparişini DC'ye taşı
  %      circshift([satır_kayma, sütun_kayma])
  F_cent    = circshift(F_p1, [-k_y, -k_x]);

  % iv) Hologram düzleminde rekonstrüksiyon alanı
  O_rec_h   = ifft2(ifftshift(F_cent));

  % v) Ters ASM: hologram düzleminden image plane'e
  O_rec_i   = asm_prop(O_rec_h, H_bwd);

  % ── Görselleştirme ───────────────────────────────────────────────
  fprintf('  [4/4] Görselleştirme...\n');
  fig = figure('visible','off','Position',[0 0 1560 1020]);

  ax1 = subplot(2,3,1);
  imagesc(x_s*1e6, x_s*1e6, angle(U_s));
  axis image; colorbar; colormap(ax1,'hsv');
  xlabel('x [µm]'); ylabel('y [µm]');
  title(sprintf('Nesne fazı – numune duzlemi\n%d µm küre | maks %.1f rad (wrap)',...
    s.cap, k0*dn*2*r), 'FontSize', 10);

  ax2 = subplot(2,3,2);
  imagesc(x_i*1e3, x_i*1e3, abs(O_holo));
  axis image; colorbar;
  xlabel('x [mm]'); ylabel('y [mm]');
  title(sprintf('Yayılmış genlik – hologram duzlemi\nFresnel difraksiyon z_h = %.0f mm',...
    z_h*1e3), 'FontSize', 10);

  ax3 = subplot(2,3,3);
  imagesc(x_i*1e3, x_i*1e3, H_proc);
  axis image; colorbar; colormap(ax3,'gray');
  xlabel('x [mm]'); ylabel('y [mm]');
  title(sprintf('Off-axis hologram – %d µm küre\npx=%.1fµm | %d-bit | saçak −45°',...
    s.cap, px_cam*1e6, bit_depth), 'FontSize', 10);

  ax4 = subplot(2,3,4);
  imagesc(log(1 + abs(FH)));
  axis image; colorbar;
  title(sprintf('|FFT(H)| – log\n+1 sipariş: sütun+%d satır+%d px', k_x, k_y),...
    'FontSize', 10);

  ax5 = subplot(2,3,5);
  imagesc(x_i*1e3, x_i*1e3, abs(O_rec_i));
  axis image; colorbar;
  xlabel('x [mm]'); ylabel('y [mm]');
  title('Rekonstr. genlik – image plane', 'FontSize', 10);

  ax6 = subplot(2,3,6);
  imagesc(x_i*1e3, x_i*1e3, angle(O_rec_i));
  axis image; colorbar; colormap(ax6,'hsv');
  xlabel('x [mm]'); ylabel('y [mm]');
  title('Rekonstr. faz [rad] – image plane', 'FontSize', 10);

  % ── Dosya kayıt ──────────────────────────────────────────────────
  f_ov  = sprintf('hologram_%s_overview.png',     s.label);
  f_raw = sprintf('hologram_%s_raw.png',           s.label);
  f_ph  = sprintf('hologram_%s_recon_phase.png',   s.label);
  f_fft = sprintf('hologram_%s_fourier_space.png', s.label);

  print(fig, f_ov, '-dpng', '-r120');

  % ── Ham hologram + rekonstrüksiyon faz (imwrite) ─────────────────
  try
    imwrite(uint16(H_cam), f_raw);
    phi = angle(O_rec_i);
    P8  = uint8(255*(phi - min(phi(:))) / (max(phi(:)) - min(phi(:))));
    imwrite(P8, f_ph);
  catch
    warning('imwrite mevcut degil.');
  end

  % ── Fourier uzayı — ayrı figür, frekans eksenli, siparişler işaretli ──
  fx_ax   = (-N/2:N/2-1) * dfx * 1e-3;        % [1/mm]
  FH_log  = log(1 + abs(FH));
  FH_vec  = sort(FH_log(:));
  nv      = numel(FH_vec);
  lo      = FH_vec(max(1,  round(0.001*nv)));
  hi      = FH_vec(min(nv, round(0.999*nv)));
  FH_clip = min(max((FH_log - lo) / (hi - lo + eps), 0), 1);

  fig_fft = figure('visible','off','Position',[0 0 860 780]);
  imagesc(fx_ax, fx_ax, FH_clip);
  axis image; colormap(gca,'hot'); colorbar;
  xlabel('f_x [1/mm]','FontSize',11);
  ylabel('f_y [1/mm]','FontSize',11);
  title(sprintf('Fourier Space — %d µm sphere\nlog|FFT(H)|   (0.1%%–99.9%% clip)', s.cap),...
    'FontSize',12);

  print(fig_fft, f_fft, '-dpng', '-r150');
  close(fig_fft);

  fprintf('  Kaydedildi: %s | %s | %s | %s\n\n', f_ov, f_raw, f_ph, f_fft);
  close(fig);
end

fprintf('=== TAMAMLANDI ===\n');

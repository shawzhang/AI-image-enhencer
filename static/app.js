/* ============================================================
   AI Image Enhancer – Client-Side Application
   ============================================================ */

(function () {
  'use strict';

  // ── State ──────────────────────────────────────────────────
  let imageId = null;
  let hasEnhanced = false;
  let isMonet = false;
  let debounceTimer = null;

  // ── DOM References ─────────────────────────────────────────
  const $ = (id) => document.getElementById(id);

  const uploadSection       = $('upload-section');
  const uploadZone          = $('upload-zone');
  const fileInput           = $('file-input');
  const comparisonSection   = $('comparison-section');
  const comparisonWrapper   = $('comparison-wrapper');
  const beforeImage         = $('before-image');
  const afterImage          = $('after-image');
  const beforePane          = $('before-pane');
  const divider             = $('comparison-divider');
  const controlsSection     = $('controls-section');
  const downloadSection     = $('download-section');
  const newUploadSection    = $('new-upload-section');
  const loadingOverlay      = $('loading-overlay');
  const enhanceBtn          = $('enhance-btn');
  const monetBtn            = $('monet-btn');
  const autoBtn             = $('auto-btn');
  const resetBtn            = $('reset-btn');
  const downloadBtn         = $('download-btn');
  const newUploadBtn        = $('new-upload-btn');
  const deviceBadge         = $('device-badge');
  const toastContainer      = $('toast-container');

  // Sliders
  const sliders = {
    noise_reduction:  { el: $('noise-reduction-slider'),  out: $('noise-reduction-value'), decimals: 0 },
    deblur_strength:  { el: $('deblur-slider'),           out: $('deblur-value'),           decimals: 1 },
    sharpness:        { el: $('sharpness-slider'),        out: $('sharpness-value'),        decimals: 1 },
    contrast:         { el: $('contrast-slider'),         out: $('contrast-value'),         decimals: 1 },
    color_saturation: { el: $('saturation-slider'),       out: $('saturation-value'),       decimals: 1 },
    brightness:       { el: $('brightness-slider'),       out: $('brightness-value'),       decimals: 0 },
    color_temp:       { el: $('color-temp-slider'),       out: $('color-temp-value'),       decimals: 0 },
  };

  // Toggles
  const superResToggle   = $('super-res-toggle');
  const superResScaleGrp = $('super-res-scale-group');
  const faceRestoreToggle = $('face-restore-toggle');

  // Accordions
  const restorationToggle  = $('restoration-toggle');
  const restorationBody    = $('restoration-controls');
  const enhancementToggle  = $('enhancement-toggle');
  const enhancementBody    = $('enhancement-controls');

  // ── Helpers ────────────────────────────────────────────────
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 3200);
  }

  function showLoading(msg) {
    $('loading-text').textContent = msg || 'Enhancing your image…';
    loadingOverlay.classList.remove('hidden');
  }

  function hideLoading() {
    loadingOverlay.classList.add('hidden');
  }

  function paintSliderTrack(slider) {
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    const val = parseFloat(slider.value);
    const pct = ((val - min) / (max - min)) * 100;

    // Determine accent color from class
    const accentMap = {
      'accent-blue':   '#5b8def',
      'accent-purple': '#a78bfa',
      'accent-teal':   '#2dd4bf',
      'accent-orange': '#fb923c',
      'accent-pink':   '#f472b6',
      'accent-yellow': '#facc15',
      'accent-red':    '#f87171',
    };

    let color = '#7c5cfc';
    for (const [cls, c] of Object.entries(accentMap)) {
      if (slider.classList.contains(cls)) { color = c; break; }
    }

    slider.style.background = `linear-gradient(to right, ${color} 0%, ${color} ${pct}%, rgba(255,255,255,0.08) ${pct}%, rgba(255,255,255,0.08) 100%)`;
  }

  function collectParams() {
    const scaleRadio = document.querySelector('input[name="super-res-scale"]:checked');
    return {
      image_id:            imageId,
      noise_reduction:     parseInt(sliders.noise_reduction.el.value, 10),
      deblur_strength:     parseFloat(sliders.deblur_strength.el.value),
      enable_super_res:    superResToggle.checked,
      super_res_scale:     scaleRadio ? scaleRadio.value : '2x',
      enable_face_restore: faceRestoreToggle.checked,
      sharpness:           parseFloat(sliders.sharpness.el.value),
      contrast:            parseFloat(sliders.contrast.el.value),
      color_saturation:    parseFloat(sliders.color_saturation.el.value),
      brightness:          parseInt(sliders.brightness.el.value, 10),
      color_temp:          parseInt(sliders.color_temp.el.value, 10),
      monet_style:         isMonet,
    };
  }

  function resetControls() {
    sliders.noise_reduction.el.value  = 0;
    sliders.deblur_strength.el.value  = 0;
    sliders.sharpness.el.value        = 0.5;
    sliders.contrast.el.value         = 1.0;
    sliders.color_saturation.el.value = 1.0;
    sliders.brightness.el.value       = 0;
    sliders.color_temp.el.value       = 0;

    superResToggle.checked  = false;
    faceRestoreToggle.checked = false;
    superResScaleGrp.classList.add('disabled');
    document.getElementById('super-res-2x').checked = true;

    isMonet = false;
    monetBtn.classList.remove('active');

    // Update all readouts & tracks
    for (const [, s] of Object.entries(sliders)) {
      s.out.textContent = parseFloat(s.el.value).toFixed(s.decimals);
      paintSliderTrack(s.el);
    }
  }

  // ── Device Detection ───────────────────────────────────────
  async function fetchDevice() {
    try {
      const res = await fetch('/api/device');
      if (!res.ok) throw new Error();
      const data = await res.json();
      deviceBadge.textContent = '⚡ ' + (data.device || 'Unknown');
    } catch {
      deviceBadge.textContent = '⚡ Device N/A';
    }
  }

  // ── Upload ─────────────────────────────────────────────────
  function handleFiles(files) {
    if (!files || files.length === 0) return;
    const file = files[0];
    const allowed = ['image/jpeg', 'image/png', 'image/heic'];
    if (!allowed.includes(file.type) && !file.name.match(/\.(jpe?g|png|heic)$/i)) {
      showToast('Unsupported file type. Use JPG, PNG, or HEIC.', 'error');
      return;
    }
    uploadFile(file);
  }

  async function uploadFile(file) {
    showLoading('Uploading image…');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Upload failed');
      }
      const data = await res.json();
      imageId = data.image_id;
      hasEnhanced = false;

      // Show original in both panes initially
      beforeImage.src = data.preview;
      afterImage.src  = data.preview;

      // Wait for image to load to set aspect ratio
      beforeImage.onload = () => {
        const aspect = beforeImage.naturalHeight / beforeImage.naturalWidth;
        comparisonWrapper.style.paddingBottom = (aspect * 100) + '%';
      };

      // Transition UI
      uploadSection.classList.add('hidden');
      comparisonSection.classList.remove('hidden');
      controlsSection.classList.remove('hidden');
      newUploadSection.classList.remove('hidden');
      downloadSection.classList.add('hidden');
      enhanceBtn.disabled = false;
      resetControls();
      resetDivider();

      showToast('Image uploaded successfully!', 'success');
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      hideLoading();
    }
  }

  // ── Enhance ────────────────────────────────────────────────
  async function enhance() {
    if (!imageId) return;
    showLoading('Enhancing your image…');
    enhanceBtn.disabled = true;

    try {
      const res = await fetch('/api/enhance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(collectParams()),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Enhancement failed');
      }

      const data = await res.json();
      afterImage.src = data.image;
      hasEnhanced = true;
      downloadSection.classList.remove('hidden');
      showToast('Enhancement complete!', 'success');
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      hideLoading();
      enhanceBtn.disabled = false;
    }
  }

  // ── Auto Analyze ───────────────────────────────────────────
  async function autoAnalyze() {
    if (!imageId) return;
    showLoading('Analyzing image…');
    autoBtn.disabled = true;

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_id: imageId }),
      });

      if (!res.ok) throw new Error('Analysis failed');

      const data = await res.json();
      if (data.error) throw new Error(data.error);

      // Apply recommended parameters to UI
      const p = data.params;
      sliders.noise_reduction.el.value = p.noise_reduction;
      sliders.deblur_strength.el.value = p.deblur_strength;
      sliders.sharpness.el.value = p.sharpness;
      sliders.contrast.el.value = p.contrast;
      sliders.color_saturation.el.value = p.color_saturation;
      sliders.brightness.el.value = p.brightness;
      sliders.color_temp.el.value = p.color_temp;

      superResToggle.checked = p.enable_super_res;
      if (p.enable_super_res) {
        superResScaleGrp.classList.remove('disabled');
        document.getElementById(`super-res-${p.super_res_scale}`).checked = true;
      } else {
        superResScaleGrp.classList.add('disabled');
      }

      faceRestoreToggle.checked = p.enable_face_restore;

      // Update all readouts & tracks
      for (const [, s] of Object.entries(sliders)) {
        s.out.textContent = parseFloat(s.el.value).toFixed(s.decimals);
        paintSliderTrack(s.el);
      }

      // Show hints
      data.hints.forEach((hint, idx) => {
        setTimeout(() => showToast(hint, 'success'), idx * 500);
      });

      // Automatically trigger enhancement
      enhance();

    } catch (err) {
      showToast(err.message, 'error');
      hideLoading();
    } finally {
      autoBtn.disabled = false;
    }
  }

  // Debounced enhance for slider changes
  function debouncedEnhance() {
    if (!imageId) return;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => enhance(), 300);
  }

  // ── Download ───────────────────────────────────────────────
  async function download() {
    if (!imageId || !hasEnhanced) return;
    const format = document.querySelector('input[name="download-format"]:checked').value;

    try {
      const res = await fetch('/api/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_id: imageId, format }),
      });

      if (!res.ok) throw new Error('Download failed');

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `enhanced.${format === 'jpeg' ? 'jpg' : 'png'}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      showToast('Download started!', 'success');
    } catch (err) {
      showToast(err.message, 'error');
    }
  }

  // ── Comparison Divider ─────────────────────────────────────
  let isDragging = false;

  function resetDivider() {
    setDividerPosition(0.5);
  }

  function setDividerPosition(ratio) {
    ratio = Math.max(0, Math.min(1, ratio));
    const pct = ratio * 100;
    beforePane.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    divider.style.left = pct + '%';
  }

  function getDividerRatio(clientX) {
    const rect = comparisonWrapper.getBoundingClientRect();
    return (clientX - rect.left) / rect.width;
  }

  function onPointerDown(e) {
    isDragging = true;
    e.preventDefault();
    setDividerPosition(getDividerRatio(e.clientX));
  }

  function onPointerMove(e) {
    if (!isDragging) return;
    e.preventDefault();
    setDividerPosition(getDividerRatio(e.clientX));
  }

  function onPointerUp() {
    isDragging = false;
  }

  // Also allow clicking anywhere on the comparison to set position
  comparisonWrapper.addEventListener('pointerdown', onPointerDown);
  document.addEventListener('pointermove', onPointerMove);
  document.addEventListener('pointerup', onPointerUp);

  // Keyboard support for divider
  divider.addEventListener('keydown', (e) => {
    const current = parseFloat(divider.style.left) / 100 || 0.5;
    const step = 0.02;
    if (e.key === 'ArrowLeft')  setDividerPosition(current - step);
    if (e.key === 'ArrowRight') setDividerPosition(current + step);
  });

  // ── Accordion ──────────────────────────────────────────────
  function setupAccordion(toggle, body) {
    toggle.addEventListener('click', () => {
      const isOpen = body.classList.contains('open');
      body.classList.toggle('open', !isOpen);
      toggle.setAttribute('aria-expanded', String(!isOpen));
    });
  }

  setupAccordion(restorationToggle, restorationBody);
  setupAccordion(enhancementToggle, enhancementBody);

  // ── Slider Events ─────────────────────────────────────────
  for (const [, s] of Object.entries(sliders)) {
    // Update readout & track on input
    s.el.addEventListener('input', () => {
      s.out.textContent = parseFloat(s.el.value).toFixed(s.decimals);
      paintSliderTrack(s.el);
    });

    // Debounced auto-enhance on change
    s.el.addEventListener('change', () => {
      if (hasEnhanced) debouncedEnhance();
    });

    // Paint initial track
    paintSliderTrack(s.el);
  }

  // ── Toggle Events ─────────────────────────────────────────
  superResToggle.addEventListener('change', () => {
    superResScaleGrp.classList.toggle('disabled', !superResToggle.checked);
  });

  // Super-res scale radios – no auto-enhance (heavy)
  // Face restore toggle – no auto-enhance (heavy)

  // ── Upload Events ─────────────────────────────────────────
  uploadZone.addEventListener('click', () => fileInput.click());
  uploadZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });

  fileInput.addEventListener('change', () => handleFiles(fileInput.files));

  // Drag & drop
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });

  uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
  });

  // ── Button Events ─────────────────────────────────────────
  enhanceBtn.addEventListener('click', enhance);
  autoBtn.addEventListener('click', autoAnalyze);
  
  monetBtn.addEventListener('click', () => {
    isMonet = !isMonet;
    monetBtn.classList.toggle('active', isMonet);
    if (hasEnhanced || isMonet) debouncedEnhance();
  });

  resetBtn.addEventListener('click', () => {
    resetControls();
    showToast('Controls reset to defaults', 'info');
  });

  downloadBtn.addEventListener('click', download);

  newUploadBtn.addEventListener('click', () => {
    imageId = null;
    hasEnhanced = false;
    fileInput.value = '';
    uploadSection.classList.remove('hidden');
    comparisonSection.classList.add('hidden');
    controlsSection.classList.add('hidden');
    downloadSection.classList.add('hidden');
    newUploadSection.classList.add('hidden');
    resetControls();
  });

  // ── Init ───────────────────────────────────────────────────
  fetchDevice();

})();

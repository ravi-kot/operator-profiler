(function () {
  function formatNum(v) {
    if (v == null || v === '') return '—';
    if (typeof v !== 'number') return String(v);
    if (Number.isInteger(v)) return v.toLocaleString();
    if (Math.abs(v) < 1e-4 || Math.abs(v) >= 1e4) return v.toExponential(2);
    return v.toFixed(2);
  }

  function render(data) {
    var llm = data.llm || {};
    var ln = data.layernorm || {};
    var subtitle = document.getElementById('subtitle');
    var cardsEl = document.getElementById('cards');
    var bodyEl = document.getElementById('table_body');
    var statusEl = document.getElementById('status');

    subtitle.textContent = data.timestamp_utc
      ? 'GPU performance summary · ' + new Date(data.timestamp_utc).toLocaleString()
      : 'GPU performance summary';

    var cardLabels = [
      ['Tokens/sec (P50)', llm.tokens_per_sec_p50],
      ['Latency P50 (s)', llm.latency_p50_s],
      ['Peak VRAM (MB)', llm.peak_vram_mb_p50],
      ['LayerNorm speedup P50', ln.speedup_p50 != null ? formatNum(ln.speedup_p50) + '×' : '—'],
      ['LayerNorm max error', ln.max_abs_error]
    ];
    cardsEl.innerHTML = cardLabels.map(function (c) {
      return '<div class="card"><div class="card_label">' + c[0] + '</div><div class="card_value">' + formatNum(c[1]) + '</div></div>';
    }).join('');

    var rows = [
      ['GPU', llm.gpu_name],
      ['LLM model', llm.model],
      ['Tokens/sec P50', llm.tokens_per_sec_p50],
      ['Latency P50 (s)', llm.latency_p50_s],
      ['Peak VRAM P50 (MB)', llm.peak_vram_mb_p50],
      ['LayerNorm baseline P50 (ms)', ln.baseline_p50_ms],
      ['LayerNorm optimized P50 (ms)', ln.optimized_p50_ms],
      ['LayerNorm speedup P50', ln.speedup_p50 != null ? formatNum(ln.speedup_p50) + '×' : null],
      ['LayerNorm max abs error', ln.max_abs_error]
    ];
    bodyEl.innerHTML = rows.map(function (r) {
      return '<tr><td>' + r[0] + '</td><td>' + formatNum(r[1]) + '</td></tr>';
    }).join('');

    statusEl.innerHTML = '<span class="status_item">LLM artifact</span><span class="status_item">LayerNorm artifact</span><span class="status_item">Summary</span>';
    statusEl.setAttribute('aria-hidden', 'false');
    cardsEl.setAttribute('aria-hidden', 'false');
    document.getElementById('table_wrap').setAttribute('aria-hidden', 'false');
  }

  var loading = document.getElementById('loading');
  var errEl = document.getElementById('error');

  function showLoading() { loading.style.display = 'block'; errEl.style.display = 'none'; }
  function showError(msg) {
    loading.style.display = 'none';
    errEl.style.display = 'block';
    errEl.textContent = 'Failed to load summary: ' + msg;
  }
  function showContent() { loading.style.display = 'none'; errEl.style.display = 'none'; }

  showLoading();
  fetch('summary.json')
    .then(function (r) {
      if (!r.ok) throw new Error(r.statusText);
      return r.json();
    })
    .then(function (data) {
      showContent();
      render(data);
    })
    .catch(function (e) {
      showError(e.message);
    });
})();

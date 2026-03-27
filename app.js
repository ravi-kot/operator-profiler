(function () {
  function formatNum(value, digits) {
    if (value == null || value === '') return '-';
    if (typeof value !== 'number') return String(value);
    if (digits != null) return value.toFixed(digits);
    if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
    return value.toFixed(2);
  }

  function card(label, value, suffix) {
    return [
      '<article class="card">',
      '<div class="card_label">', label, '</div>',
      '<div class="card_value">', formatNum(value), suffix || '', '</div>',
      '</article>'
    ].join('');
  }

  function renderSummary(data) {
    var headline = data.headline || {};
    var burstRows = ((data.experiments || {}).burst_load) || [];
    var blockRows = ((data.experiments || {}).block_sweep) || [];
    var chunkRows = ((data.experiments || {}).chunked_prefill) || [];
    var quantRows = ((data.experiments || {}).kv_quantization) || [];
    var takeaways = data.key_findings || [];

    document.getElementById('subtitle').textContent =
      (data.project && data.project.tagline ? data.project.tagline + ' ' : '') +
      'Target: ' + (headline.gpu_target || 'workstation GPU') +
      '. Runtime used here: ' + (headline.runtime_device || 'unknown') + '.';

    document.getElementById('status').innerHTML = [
      '<span class="pill">Hot Prefix Pinning</span>',
      '<span class="pill">Request-Aware Policy</span>',
      '<span class="pill">Reuse-Aware Scheduler</span>',
      '<span class="pill">Cost-Aware Mode</span>',
      '<span class="pill">CUDA Included</span>'
    ].join('');

    document.getElementById('cards').innerHTML = [
      card('TTFT P50', headline.ttft_ms_p50, ' ms'),
      card('Decode Latency / Token', headline.decode_latency_ms_per_token_p50, ' ms'),
      card('Throughput', headline.throughput_tokens_per_s, ' tok/s'),
      card('Peak KV Memory', headline.peak_kv_memory_mb, ' MB'),
      card('Cache Hit Rate', (headline.cache_hit_rate || 0) * 100, '%'),
      card('Prefix Reuse Rate', (headline.prefix_reuse_rate || 0) * 100, '%')
    ].join('');

    document.getElementById('takeaways').innerHTML = takeaways.map(function (text) {
      return '<div class="takeaway">' + text + '</div>';
    }).join('');

    document.getElementById('burst_body').innerHTML = burstRows.map(function (row) {
      return [
        '<tr>',
        '<td>', row.concurrency, '</td>',
        '<td>', formatNum(row.baseline.ttft_ms.p50), ' ms</td>',
        '<td>', formatNum(row.advanced.ttft_ms.p50), ' ms</td>',
        '<td>', formatNum(row.baseline.throughput_tokens_per_s), '</td>',
        '<td>', formatNum(row.advanced.throughput_tokens_per_s), '</td>',
        '<td>', formatNum((row.advanced.cache_hit_rate || 0) * 100), '%</td>',
        '</tr>'
      ].join('');
    }).join('');

    document.getElementById('block_body').innerHTML = blockRows.map(function (row) {
      return [
        '<tr>',
        '<td>', row.block_size_tokens, '</td>',
        '<td>', formatNum(row.throughput_tokens_per_s), '</td>',
        '<td>', formatNum((row.prefix_reuse_rate || 0) * 100), '%</td>',
        '<td>', formatNum(row.fragmentation_ratio, 4), '</td>',
        '</tr>'
      ].join('');
    }).join('');

    document.getElementById('chunk_body').innerHTML = chunkRows.map(function (row) {
      var chunkLabel = row.chunked_prefill_tokens === 0 ? 'off' : row.chunked_prefill_tokens;
      return [
        '<tr>',
        '<td>', chunkLabel, '</td>',
        '<td>', formatNum(row.ttft_ms_p50), ' ms</td>',
        '<td>', formatNum(row.ttft_ms_p95), ' ms</td>',
        '<td>', formatNum(row.throughput_tokens_per_s), '</td>',
        '</tr>'
      ].join('');
    }).join('');

    document.getElementById('quant_body').innerHTML = quantRows.map(function (row) {
      return [
        '<tr>',
        '<td>', row.kv_mode, '</td>',
        '<td>', formatNum(row.peak_kv_memory_mb), ' MB</td>',
        '<td>', formatNum(row.throughput_tokens_per_s), '</td>',
        '<td>', formatNum(row.cosine_similarity, 6), '</td>',
        '<td>', formatNum(row.rmse, 4), '</td>',
        '</tr>'
      ].join('');
    }).join('');
  }

  if (window.__KV_CACHE_SUMMARY__) {
    renderSummary(window.__KV_CACHE_SUMMARY__);
    return;
  }

  fetch('summary.json')
    .then(function (response) {
      if (!response.ok) {
        throw new Error(response.statusText);
      }
      return response.json();
    })
    .then(renderSummary)
    .catch(function (error) {
      document.getElementById('subtitle').textContent =
        'Could not load summary.json. Open the dashboard through a local server or regenerate dashboard/summary.js. ' + error.message;
    });
})();

window.__KV_CACHE_SUMMARY__ = {
  "timestamp_utc": "2026-03-27T04:01:41.326038+00:00",
  "project": {
    "name": "KV Cache Serving Lab",
    "tagline": "Paged KV cache, hot-prefix pinning, request-aware policy, reuse-aware scheduling, and cost-aware memory control."
  },
  "headline": {
    "gpu_target": "RTX 5070-class workstation target with 8 GB total VRAM budget",
    "runtime_device": "CPU fallback in this environment",
    "ttft_ms_p50": 2223.8588000000045,
    "decode_latency_ms_per_token_p50": 4.740183783783779,
    "throughput_tokens_per_s": 1901.5452858837143,
    "peak_kv_memory_mb": 12460.78125,
    "average_active_kv_pages": 1808.9651678277392,
    "cache_hit_rate": 0.9891059353869271,
    "prefix_reuse_rate": 0.9891059353869271,
    "evictions": 0,
    "oom_avoided": 0,
    "fragmentation_ratio": 0.0031570135596812265,
    "block_size_tokens": 16
  },
  "experiments": {
    "burst_load": [
      {
        "concurrency": 10,
        "baseline": {
          "ttft_ms": {
            "p50": 408.2472080000005,
            "p95": 1728.0476079999958,
            "mean": 576.8293075999998,
            "max": 1730.0476079999958
          },
          "decode_latency_ms_per_token": {
            "p50": 5.78086956521738,
            "p95": 6.248235294117641,
            "mean": 4.94873086385006,
            "max": 6.279999999999984
          },
          "throughput_tokens_per_s": 1655.2976883516583,
          "peak_kv_memory_mb": 4856.71875,
          "average_active_kv_pages": 883.6228748068006,
          "cache_hit_rate": 0.8914285714285715,
          "prefix_reuse_rate": 0.8914285714285715,
          "evictions": 0,
          "oom_avoided": 0,
          "oom_failures": 0,
          "fragmentation_ratio": 0.004835205634192174,
          "queue_delay_ms": {
            "p50": 94.44320000000056,
            "p95": 1702.8972079999958,
            "mean": 321.0391411999998,
            "max": 1704.8972079999958
          },
          "completed_requests": 20,
          "failed_requests": 0
        },
        "advanced": {
          "ttft_ms": {
            "p50": 347.3439599999992,
            "p95": 1735.5064399999997,
            "mean": 524.1307339999993,
            "max": 1775.5064399999997
          },
          "decode_latency_ms_per_token": {
            "p50": 5.692075362318841,
            "p95": 6.285618181818186,
            "mean": 4.937582306008529,
            "max": 6.302800000000004
          },
          "throughput_tokens_per_s": 1684.3174123606486,
          "peak_kv_memory_mb": 4905.46875,
          "average_active_kv_pages": 875.1002994011976,
          "cache_hit_rate": 0.9447619047619048,
          "prefix_reuse_rate": 0.9447619047619048,
          "evictions": 0,
          "oom_avoided": 0,
          "oom_failures": 0,
          "fragmentation_ratio": 0.004391863327361589,
          "queue_delay_ms": {
            "p50": 88.89919999999921,
            "p95": 1716.3435599999996,
            "mean": 313.7208939999993,
            "max": 1756.3435599999996
          },
          "completed_requests": 20,
          "failed_requests": 0
        },
        "advanced_scheduler": {
          "batches": 3.0,
          "reuse_priority_batches": 1.0,
          "reduced_batch_events": 0.0
        },
        "advanced_cache": {
          "resident_pages": 114.0,
          "current_memory_mb": 285.0,
          "peak_memory_mb": 4905.46875,
          "cache_hit_rate": 0.9447619047619048,
          "prefix_reuse_rate": 0.9447619047619048,
          "evictions": 0.0,
          "oom_avoided": 0.0,
          "oom_failures": 0.0,
          "fragmentation_ratio": 0.004391863327361589,
          "pinned_pages": 56.0
        }
      },
      {
        "concurrency": 50,
        "baseline": {
          "ttft_ms": {
            "p50": 5544.351081600033,
            "p95": 8069.079190399991,
            "mean": 4508.447356384009,
            "max": 9154.590390400013
          },
          "decode_latency_ms_per_token": {
            "p50": 5.400212499999995,
            "p95": 7.194875609756097,
            "mean": 5.319963847329017,
            "max": 7.416399999999994
          },
          "throughput_tokens_per_s": 1762.8578632720569,
          "peak_kv_memory_mb": 7224.53125,
          "average_active_kv_pages": 1407.7666666666667,
          "cache_hit_rate": 0.9784743202416919,
          "prefix_reuse_rate": 0.9784743202416919,
          "evictions": 0,
          "oom_avoided": 0,
          "oom_failures": 0,
          "fragmentation_ratio": 0.0033655469478186184,
          "queue_delay_ms": {
            "p50": 5172.703945600033,
            "p95": 7781.838681599991,
            "mean": 4235.92303792001,
            "max": 9145.297590400014
          },
          "completed_requests": 100,
          "failed_requests": 0
        },
        "advanced": {
          "ttft_ms": {
            "p50": 2873.0513903999945,
            "p95": 8038.496692799924,
            "mean": 3149.541914655975,
            "max": 8113.496692799924
          },
          "decode_latency_ms_per_token": {
            "p50": 4.4504800000000015,
            "p95": 7.553992156862744,
            "mean": 4.906084729278044,
            "max": 7.843600000000001
          },
          "throughput_tokens_per_s": 1933.1442570600632,
          "peak_kv_memory_mb": 8782.8125,
          "average_active_kv_pages": 1452.842737722048,
          "cache_hit_rate": 0.9890483383685801,
          "prefix_reuse_rate": 0.9890483383685801,
          "evictions": 0,
          "oom_avoided": 0,
          "oom_failures": 0,
          "fragmentation_ratio": 0.004373516401899107,
          "queue_delay_ms": {
            "p50": 2597.9747999999945,
            "p95": 8005.542652799924,
            "mean": 3052.2680532479744,
            "max": 8080.542652799924
          },
          "completed_requests": 100,
          "failed_requests": 0
        },
        "advanced_scheduler": {
          "batches": 8.0,
          "reuse_priority_batches": 6.0,
          "reduced_batch_events": 0.0
        },
        "advanced_cache": {
          "resident_pages": 114.0,
          "current_memory_mb": 285.0,
          "peak_memory_mb": 8782.8125,
          "cache_hit_rate": 0.9890483383685801,
          "prefix_reuse_rate": 0.9890483383685801,
          "evictions": 0.0,
          "oom_avoided": 0.0,
          "oom_failures": 0.0,
          "fragmentation_ratio": 0.004373516401899107,
          "pinned_pages": 56.0
        }
      },
      {
        "concurrency": 100,
        "baseline": {
          "ttft_ms": {
            "p50": 10963.677040000002,
            "p95": 18880.514719999846,
            "mean": 10026.789340064002,
            "max": 20544.040623999983
          },
          "decode_latency_ms_per_token": {
            "p50": 5.80557560975611,
            "p95": 7.059647058823528,
            "mean": 5.648114562626344,
            "max": 7.365999999999989
          },
          "throughput_tokens_per_s": 1612.8367828770204,
          "peak_kv_memory_mb": 7119.0625,
          "average_active_kv_pages": 1798.8575991482567,
          "cache_hit_rate": 0.9893218433870363,
          "prefix_reuse_rate": 0.9893218433870363,
          "evictions": 0,
          "oom_avoided": 0,
          "oom_failures": 0,
          "fragmentation_ratio": 0.0029228571823298696,
          "queue_delay_ms": {
            "p50": 10601.099384000003,
            "p95": 18494.114563199848,
            "mean": 9740.512493344002,
            "max": 20376.701519999984
          },
          "completed_requests": 200,
          "failed_requests": 0
        },
        "advanced": {
          "ttft_ms": {
            "p50": 6987.999614400048,
            "p95": 17765.031795200255,
            "mean": 7647.505390592068,
            "max": 17813.031795200255
          },
          "decode_latency_ms_per_token": {
            "p50": 4.655675675675686,
            "p95": 6.9339999999999895,
            "mean": 4.996228793372361,
            "max": 7.23400000000001
          },
          "throughput_tokens_per_s": 1816.9819031242032,
          "peak_kv_memory_mb": 9339.6875,
          "average_active_kv_pages": 1793.9952843273231,
          "cache_hit_rate": 0.9945672536530535,
          "prefix_reuse_rate": 0.9945672536530535,
          "evictions": 0,
          "oom_avoided": 0,
          "oom_failures": 0,
          "fragmentation_ratio": 0.0029756303741384593,
          "queue_delay_ms": {
            "p50": 6732.838080000048,
            "p95": 17461.786030400253,
            "mean": 7537.012948256066,
            "max": 17509.786030400253
          },
          "completed_requests": 200,
          "failed_requests": 0
        },
        "advanced_scheduler": {
          "batches": 15.0,
          "reuse_priority_batches": 13.0,
          "reduced_batch_events": 0.0
        },
        "advanced_cache": {
          "resident_pages": 114.0,
          "current_memory_mb": 285.0,
          "peak_memory_mb": 9339.6875,
          "cache_hit_rate": 0.9945672536530535,
          "prefix_reuse_rate": 0.9945672536530535,
          "evictions": 0.0,
          "oom_avoided": 0.0,
          "oom_failures": 0.0,
          "fragmentation_ratio": 0.0029756303741384593,
          "pinned_pages": 56.0
        }
      }
    ],
    "block_sweep": [
      {
        "block_size_tokens": 8,
        "throughput_tokens_per_s": 1111.2770229913162,
        "cache_hit_rate": 0.9918835712286594,
        "prefix_reuse_rate": 0.9918835712286594,
        "fragmentation_ratio": 0.001212616234841078,
        "peak_kv_memory_mb": 17431.5625,
        "decode_latency_ms_per_token_p50": 5.291881927710849
      },
      {
        "block_size_tokens": 16,
        "throughput_tokens_per_s": 1944.4304123213835,
        "cache_hit_rate": 0.9916112236042812,
        "prefix_reuse_rate": 0.9916112236042812,
        "fragmentation_ratio": 0.002820427842144759,
        "peak_kv_memory_mb": 16954.375,
        "decode_latency_ms_per_token_p50": 4.491984210526307
      },
      {
        "block_size_tokens": 32,
        "throughput_tokens_per_s": 2211.863747254754,
        "cache_hit_rate": 0.9918125352907962,
        "prefix_reuse_rate": 0.9918125352907962,
        "fragmentation_ratio": 0.007403108789965699,
        "peak_kv_memory_mb": 18220.46875,
        "decode_latency_ms_per_token_p50": 4.511023255813961
      },
      {
        "block_size_tokens": 64,
        "throughput_tokens_per_s": 2281.875345446374,
        "cache_hit_rate": 0.9913842619184376,
        "prefix_reuse_rate": 0.9913842619184376,
        "fragmentation_ratio": 0.013980249858352256,
        "peak_kv_memory_mb": 15452.03125,
        "decode_latency_ms_per_token_p50": 4.494273170731714
      }
    ],
    "chunked_prefill": [
      {
        "chunked_prefill_tokens": 0,
        "ttft_ms_p50": 7417.056533759988,
        "ttft_ms_p95": 12693.031041599885,
        "throughput_tokens_per_s": 951.7509478725832,
        "oom_avoided": 0,
        "oom_failures": 0,
        "peak_kv_memory_mb": 6164.53125
      },
      {
        "chunked_prefill_tokens": 256,
        "ttft_ms_p50": 6842.090812400113,
        "ttft_ms_p95": 11921.280869199898,
        "throughput_tokens_per_s": 1100.6920611333492,
        "oom_avoided": 0,
        "oom_failures": 0,
        "peak_kv_memory_mb": 7654.53125
      },
      {
        "chunked_prefill_tokens": 512,
        "ttft_ms_p50": 6553.681100799954,
        "ttft_ms_p95": 12149.901660799995,
        "throughput_tokens_per_s": 1091.8906277418769,
        "oom_avoided": 0,
        "oom_failures": 0,
        "peak_kv_memory_mb": 7752.03125
      },
      {
        "chunked_prefill_tokens": 1024,
        "ttft_ms_p50": 7090.16090079992,
        "ttft_ms_p95": 12931.31082080001,
        "throughput_tokens_per_s": 1044.2912859557066,
        "oom_avoided": 0,
        "oom_failures": 0,
        "peak_kv_memory_mb": 7428.4375
      }
    ],
    "kv_quantization": [
      {
        "kv_mode": "fp16",
        "peak_kv_memory_mb": 14831.875,
        "throughput_tokens_per_s": 1742.6984385687808,
        "decode_latency_ms_per_token_p50": 4.552000000000006,
        "cache_hit_rate": 0.9904037061548643,
        "cosine_similarity": 0.9999988079071045,
        "rmse": 0.00020850994042120874,
        "max_abs_error": 0.001800537109375
      },
      {
        "kv_mode": "fp8",
        "peak_kv_memory_mb": 7415.9375,
        "throughput_tokens_per_s": 1795.6304598457539,
        "decode_latency_ms_per_token_p50": 4.40992,
        "cache_hit_rate": 0.9904037061548643,
        "cosine_similarity": 0.9999595880508423,
        "rmse": 0.008984656073153019,
        "max_abs_error": 0.021801233291625977
      },
      {
        "kv_mode": "int8",
        "peak_kv_memory_mb": 7415.9375,
        "throughput_tokens_per_s": 1795.6304598457539,
        "decode_latency_ms_per_token_p50": 4.40992,
        "cache_hit_rate": 0.9904037061548643,
        "cosine_similarity": 0.9999762177467346,
        "rmse": 0.006823353469371796,
        "max_abs_error": 0.016453146934509277
      }
    ]
  },
  "key_findings": [
    "At 100 concurrent requests, reuse-aware serving cut TTFT p50 from 10964 ms to 6988 ms and raised throughput from 1613 to 1817 tok/s.",
    "Block size 64 delivered the best throughput in the sweep while exposing the reuse versus fragmentation tradeoff.",
    "fp8 KV mode cut peak KV memory by about 50.0% versus fp16 in the quantization study."
  ]
};

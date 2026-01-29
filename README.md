# OperatorProfiler

GPU operator performance : LLM inference benchmark, LayerNorm (Triton), and a summary dashboard.

**Upload to GitHub:** see [GITHUB.md](GITHUB.md) for step-by-step instructions.

## Benchmarks

- **LLM inference**: `python -m bench.llm_infer_bench` → `artifacts/llm.json`
- **LayerNorm**: `python -m bench.layernorm_bench` → `artifacts/layernorm.json`
- **Summary**: `python -m bench.summarize` → `artifacts/summary.json` and `summary.csv`

Use conda env `ml` and run from repo root.

## Dashboard (Vercel, no npm)

Static HTML/CSS/JS dashboard. **No Node.js or npm** — no install, no build.

1. **Update data**: copy `artifacts/summary.json` into `dashboard/summary.json`.
2. **Local**: open `dashboard/index.html` in a browser, or run `cd dashboard && python -m http.server 8080 --bind 127.0.0.1` and open **http://127.0.0.1:8080**.
3. **Deploy**: Push to GitHub, then in [Vercel](https://vercel.com) import the repo and set **Root Directory** to `dashboard`. No build step.

See `dashboard/README.md` for details.

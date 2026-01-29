# OperatorProfiler Dashboard (static, no npm)

Plain HTML/CSS/JS dashboard. **No Node.js or npm** — no install, no `node_modules`, no build.

## Local

1. **Update data**: copy `../artifacts/summary.json` into this folder as `summary.json` (overwrite).
2. **View**:
   - **Option A**: Open `index.html` in a browser. If the page says "Failed to load summary", use Option B.
   - **Option B**: From this folder run `python -m http.server 8080 --bind 127.0.0.1`, then open **http://127.0.0.1:8080** (needs Python only).

## Deploy to Vercel

1. Push the repo to GitHub.
2. In [Vercel](https://vercel.com): **Add New Project** → import this repo.
3. Set **Root Directory** to `dashboard`.
4. **Build Command**: leave empty or set to `echo done` (no build).
5. **Output Directory**: leave empty; Vercel will serve the folder as static files.
6. Deploy. Reviewers get a URL like `https://your-project.vercel.app`.

Before deploying, copy `artifacts/summary.json` into `dashboard/summary.json` so the live site shows your latest metrics.

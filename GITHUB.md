# Upload this project to GitHub

Do these steps in a terminal (PowerShell or Command Prompt) from the **project root** `operator-profiler`.

## 1. Install Git (if needed)

If `git` is not recognized:

- Download: https://git-scm.com/download/win  
- Run the installer and restart the terminal.

## 2. Create a new repo on GitHub

1. Go to https://github.com/new  
2. **Repository name**: `operator-profiler` (or any name you like)  
3. **Public**, no README, no .gitignore (we already have them)  
4. Click **Create repository**

## 3. Initialize and push from your machine

In the project folder run:

```powershell
cd C:\Users\Admin\Workspace\operator-profiler

git init
git add .
git status
git commit -m "OperatorProfiler: LLM bench, LayerNorm (Triton), summarize, static dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/operator-profiler.git
git push -u origin main
```

Replace **YOUR_USERNAME** with your GitHub username (and **operator-profiler** with your repo name if different).

If GitHub asks for auth, use a **Personal Access Token** (Settings → Developer settings → Personal access tokens) as the password, or sign in with GitHub CLI.

## 4. After the first push

- **Dashboard**: The repo already has `dashboard/summary.json` so Vercel will show data. To refresh it later: copy `artifacts/summary.json` to `dashboard/summary.json`, commit, and push.
- **Artifacts**: The root folder `artifacts/` is in `.gitignore` so it is not pushed. To commit a demo snapshot, remove the `artifacts/` line from `.gitignore`, then `git add artifacts/` and commit.

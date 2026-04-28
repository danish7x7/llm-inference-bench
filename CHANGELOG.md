# LLM Inference Benchmarking Suite — Changelog

Daily log of work done. **Format:** Each entry has Date / What I Did / What Worked / What Broke / Tomorrow's First Task.

This file is for two purposes:
1. Continuity — when I pick this back up after a break, the latest entry tells me exactly where I left off
2. AI context — when I start a new chat with Claude, I paste the latest 1–3 entries so it has full context

---

## 2026-04-26 — Day 1 (Saturday)

**What I did:**
- Verified WSL2 Ubuntu setup, NVIDIA RTX 4060 detected via `nvidia-smi` (CUDA 12.7, driver 566.24)
- Major C: drive cleanup: deleted Anaconda3 (6.4GB), Python 3.13 standalone, Python 3.12 Microsoft Store install, disabled Python execution aliases
- Cleared pip cache, emptied Recycle Bin
- C: free space went from 22.8GB → ~50GB
- Uninstalled Wuthering Waves from D: drive — D: free space 97GB → 248GB
- Installed Miniconda inside WSL Ubuntu (`conda 26.1.1`, Python 3.13.12 base)
- Created `~/projects/llm-inference-bench/` with `.gitignore`

**What worked:**
- All cleanup steps completed without breaking Windows
- Miniconda install in WSL clean — `(base)` prompt active
- WSL has 953GB virtual filesystem available

**What broke:**
- Drivers folder still exists (5.6GB) — chose to keep, not blocking
- Initial confusion about Python 3.13 + vLLM compatibility — verified vLLM 0.19+ supports 3.13

**Tomorrow's first task (Day 2):**
- `cd ~/projects/llm-inference-bench`
- `conda create -n vllm python=3.13 -y && conda activate vllm`
- `pip install vllm`
- Run hello world test with OPT-125M model
- Commit everything to git

**Open decisions:**
- Stay on C: drive WSL or migrate to D: → decided STAY ON C: for now (sufficient space)
- Python 3.13 vs 3.11 → decided 3.13 (verified vLLM supports it)

---

## (Template for future entries)

### YYYY-MM-DD — Day N (Day of week)

**What I did:**
-

**What worked:**
-

**What broke:**
-

**Tomorrow's first task:**
-

**Open decisions:**
-

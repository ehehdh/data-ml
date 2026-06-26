# Complaint Importance Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Korean civil complaint pipeline that flattens AI Hub data, clusters similar complaints, extracts representative samples, labels them through Gemini/Gemma batch jobs, and exports CSVs for Colab or Orange3 training.

**Architecture:** Keep the project root flat and script-driven. Split reusable pipeline behavior into testable functions while preserving a CLI entrypoint for local and Colab execution. Use Gemini Batch API JSONL input for large LLM labeling so the project avoids one-request-at-a-time labeling for normal runs.

**Tech Stack:** Python 3.10+, pandas, scikit-learn, scipy, google-genai, python-dotenv, pytest.

---

### Task 1: Clean Workspace Structure

**Files:**
- Move: `데이터과학과 머신러닝/데이터과학과 머신러닝/*` to project root
- Delete: broken `.venv`, temporary `outputs/smoke_*`, empty `.agents`, empty `.codex`
- Modify: `.gitignore`

- [ ] **Step 1: Verify cleanup targets**

Run: `Get-ChildItem -Force` from the repository root and confirm the duplicate nested directory exists.
Expected: project files live under `데이터과학과 머신러닝/데이터과학과 머신러닝`.

- [ ] **Step 2: Move inner project files to root**

Move each child from the inner project directory to the repository root, then remove the now-empty wrapper directory.
Expected: `scripts`, `config`, `requirements.txt`, `.env.example`, `README_pipeline.md`, `outputs`, and the raw dataset directory are direct children of the repository root.

- [ ] **Step 3: Delete generated or broken local artifacts**

Remove only the broken `.venv`, `outputs/smoke_*`, and empty Codex-local directories.
Expected: raw source zips and `outputs/complaints_flat.csv` remain intact.

### Task 2: Add Tests Around Labeling Batch Contracts

**Files:**
- Create: `tests/test_pipeline_batch.py`
- Modify: `requirements-dev.txt`

- [ ] **Step 1: Write failing tests for batch request JSONL**

Test that batch request records contain stable request metadata, prompt contents, JSON MIME config, and one request per target row.
Expected: tests fail because batch helper functions are not implemented yet.

- [ ] **Step 2: Write failing tests for batch result parsing**

Test successful JSON responses, per-row API errors, and invalid model JSON handling.
Expected: tests fail because parser functions are not implemented yet.

### Task 3: Refactor Pipeline for Batch Labeling

**Files:**
- Modify: `scripts/pipeline.py`

- [ ] **Step 1: Implement request builders**

Add pure functions that turn target CSV rows into Gemini Batch API GenerateContent request dictionaries and JSONL lines.
Expected: request builder tests pass.

- [ ] **Step 2: Implement result parsers**

Add pure functions that read batch JSONL responses and write `gemma_labels.csv` plus `gemma_failed.jsonl`.
Expected: parser tests pass.

- [ ] **Step 3: Add CLI commands**

Add `label-batch-create`, `label-batch-status`, and `label-batch-collect` commands. Keep existing `label` as a small synchronous fallback.
Expected: CLI help lists the new commands.

### Task 4: Verify End-to-End Smoke Flow

**Files:**
- Modify: `README_pipeline.md`

- [ ] **Step 1: Run automated tests**

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Run non-API smoke flow**

Run flatten, cluster, sample, and batch JSONL preparation with a small limit.
Expected: smoke files are created and batch JSONL is valid without calling the API.

- [ ] **Step 3: Update documentation**

Document local setup, Colab path, Orange3 export path, synchronous label fallback, and batch labeling workflow.
Expected: README gives copy-paste commands for the normal project path.

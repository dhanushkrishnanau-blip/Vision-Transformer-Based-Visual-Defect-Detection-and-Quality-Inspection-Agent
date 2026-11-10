# Project rules for Copilot

Project: Vision Transformer-Based Visual Defect Detection and Quality Inspection Agent (college capstone, "Project 12"). The owner is a beginner: explain things simply, comment the code, keep files small and readable.

Architecture:
- The detector (DINOv2 + FAISS, folder detector/) runs ONLY on Google Colab. It produces demo_data/detections.csv (image_path, heatmap_path, score, decision, timestamp), heatmaps in demo_data/heatmaps/, and demo_data/thresholds.json. Never run or import detector code on this laptop.
- The dashboard is now a React + Vite frontend (folder frontend/) and a FastAPI backend (folder backend/). Leave the legacy app/ folder untouched.
- The agent (folder agent/), the backend (folder backend/), and the tests (folder tests/) run on this Windows laptop inside the .venv.
- The database is Supabase (hosted Postgres), accessed over its REST API (/rest/v1/<table>) with the requests library. Tables already exist, do not create or alter them: inspections(id, image_path unique, heatmap_path, score, decision, created_at), tickets(id, inspection_id, defect_type, severity, procedure_text, status open/in_progress/closed, created_at), reviews(id, inspection_id, reason, reviewer_label, notes, created_at, reviewed_at), decision_log(id, inspection_id, step, tool_name, input_summary, output_summary, created_at).

Environment limits: a Windows Application Control policy blocks packages with compiled DLLs. torch and pyarrow are blocked, so pandas does not work either. In agent/ and the legacy app/ folder use only: the Python standard library, requests, python-dotenv, langgraph, pytest, and streamlit (only if "import streamlit" works). In backend/ and tests/ also allow fastapi, plain uvicorn (never uvicorn[standard]), httpx, and pytest. Never import pandas, numpy, pyarrow, torch, cv2, scikit-learn or the supabase python package in agent/, app/, backend/, or tests/. Pillow only inside try/except with a fallback. For the legacy Streamlit app, never use st.dataframe, st.table or st.bar_chart (they need pandas/pyarrow); render tables and bars as simple HTML. Show images as base64 HTML <img>, not st.image. Always use the .venv Python.
- The frontend must never contain or receive Supabase keys. All Supabase access goes through the backend using agent/db.py.
- The backend may serve image files only from data/mvtec and demo_data. Resolve and verify every requested path is inside its allowed folder to prevent path traversal.

Secrets: SUPABASE_URL and SUPABASE_SECRET_KEY live in .env, which is gitignored. Never open, print, log, copy or commit .env. Never hard-code keys. Send the key in the "apikey" header only. Never use "git add -f". Before any commit, run git status and make sure .env, data/, results/ and .venv/ are not staged; if they are, stop and tell me.

Working style: when I ask for a phase, do all of it: write the code, install missing packages in the .venv, run the tests, fix any failures yourself, and finish with a short summary: what changed, how you tested it, and what I must do by hand. Do not run git push unless the prompt says so.

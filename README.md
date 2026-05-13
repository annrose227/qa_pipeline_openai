# Customer Support QA Pipeline (OpenAI GPT-4.1)

Powered by **OpenAI GPT-4.1**.

## Quick Start

### 1. Get an OpenAI API key

Go to https://platform.openai.com/api-keys and create a key.

### 2. Set your API key

**macOS / Linux:**

```bash
export OPENAI_API_KEY=sk-proj-your-key-here
```

**Windows PowerShell:**

```powershell
$env:OPENAI_API_KEY = "sk-proj-your-key-here"
```

**Windows Command Prompt:**

```cmd
set OPENAI_API_KEY=sk-proj-your-key-here
```

### 3. Run the pipeline

```bash
pip install requests
python pipeline_openai.py
```

Or with make (macOS/Linux):

```bash
make run
```

## All Commands

```bash
python pipeline_openai.py   # full pipeline run
python validate.py          # validate generated outputs only
```

## Pipeline Stages

```
INIT
 -> INPUTS_LOADED          (qa_framework.json + transcripts/*.txt)
 -> TRANSCRIPTS_PARSED     (deterministic, no LLM)
 -> QA_SCORED              (Stage 1 — one GPT-4.1 call per transcript)
 -> COMPLIANCE_EXTRACTED   (Stage 2 — one GPT-4.1 call per transcript)
 -> COACHING_GENERATED     (Stage 3 — one GPT-4.1 call, NO QA scores passed)
 -> DASHBOARD_COMPUTED     (deterministic grade calculation in code)
 -> VALIDATION_COMPLETE    (automated checks)
 -> RESULTS_FINALISED
```

## Total LLM Calls: 11

- 3 × QA Scoring (one per transcript)
- 3 × Compliance Extraction (one per transcript)
- 3 × Coaching Notes (one per transcript)
- 1 × Calibration Check
- 1 × Rebuttal Coaching (T-1042)

## Output Files

| File                        | Description                                 |
| --------------------------- | ------------------------------------------- |
| `parsed_transcripts/*.json` | Structured turn-by-turn records             |
| `qa_scores.json`            | Per-dimension scores + weighted totals      |
| `compliance_flags.json`     | Risk flags with severity                    |
| `coaching_notes.md`         | Personalised coaching notes                 |
| `dashboard_summary.md`      | Summary table with grades (A/B/C/D/F)       |
| `calibration_check.json`    | Cross-transcript scoring consistency        |
| `team_trend.json`           | Delta vs historical baseline                |
| `escalation_cases.json`     | Auto-fail + critical compliance escalations |
| `rebuttal_coaching.md`      | De-escalation role-play scenarios           |
| `llm_calls.jsonl`           | Full LLM call audit log                     |

## Replacing Transcripts / Framework

- Replace `qa_framework.json` with your own (same JSON structure)
- Add/replace `.txt` files in `transcripts/` (format: `--- CALL_ID (Agent: NAME) ---`)
- Delete generated files and re-run: `python pipeline_openai.py`

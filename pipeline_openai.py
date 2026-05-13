#!/usr/bin/env python3
"""
QA Pipeline — multi-stage AI quality assurance for customer support transcripts.
Powered by OpenAI GPT-4.1.

Stages:
  INIT -> INPUTS_LOADED -> TRANSCRIPTS_PARSED -> QA_SCORED ->
  COMPLIANCE_EXTRACTED -> COACHING_GENERATED -> DASHBOARD_COMPUTED ->
  VALIDATION_COMPLETE -> RESULTS_FINALISED
"""

import hashlib
import json
import os
import re
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config — OpenAI
# ---------------------------------------------------------------------------
OPENAI_MODEL = "gpt-4.1"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MAX_TOKENS = 4096

ROOT = Path(__file__).parent
TRANSCRIPTS_DIR = ROOT / "transcripts"
PARSED_DIR = ROOT / "parsed_transcripts"
QA_FRAMEWORK_FILE = ROOT / "qa_framework.json"
QA_SCORES_FILE = ROOT / "qa_scores.json"
COMPLIANCE_FLAGS_FILE = ROOT / "compliance_flags.json"
COACHING_NOTES_FILE = ROOT / "coaching_notes.md"
DASHBOARD_FILE = ROOT / "dashboard_summary.md"
CALIBRATION_FILE = ROOT / "calibration_check.json"
TEAM_TREND_FILE = ROOT / "team_trend.json"
ESCALATION_FILE = ROOT / "escalation_cases.json"
REBUTTAL_FILE = ROOT / "rebuttal_coaching.md"
LLM_LOG_FILE = ROOT / "llm_calls.jsonl"

# ---------------------------------------------------------------------------
# Grade thresholds (deterministic, not delegated to LLM)
# ---------------------------------------------------------------------------
GRADE_THRESHOLDS = [
    (85, "A"),
    (70, "B"),
    (55, "C"),
    (40, "D"),
]


def compute_grade(weighted_score: float, auto_fail: bool) -> str:
    if auto_fail:
        return "F"
    for threshold, grade in GRADE_THRESHOLDS:
        if weighted_score >= threshold:
            return grade
    return "F"


# ---------------------------------------------------------------------------
# Pipeline state machine
# ---------------------------------------------------------------------------
STAGES = [
    "INIT",
    "INPUTS_LOADED",
    "TRANSCRIPTS_PARSED",
    "QA_SCORED",
    "COMPLIANCE_EXTRACTED",
    "COACHING_GENERATED",
    "DASHBOARD_COMPUTED",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
]


class PipelineState:
    def __init__(self):
        self.stage = "INIT"
        self.transcripts: list[dict] = []
        self.parsed: dict[str, dict] = {}
        self.qa_scores: dict[str, dict] = {}
        self.compliance_flags: dict[str, list] = {}
        self.coaching_notes: dict[str, str] = {}
        self.framework: dict = {}
        self.llm_log: list[dict] = []

    def advance(self, next_stage: str):
        idx_current = STAGES.index(self.stage)
        idx_next = STAGES.index(next_stage)
        if idx_next != idx_current + 1:
            raise RuntimeError(
                f"Invalid stage transition: {self.stage} -> {next_stage}"
            )
        print(f"  [stage] {self.stage} -> {next_stage}")
        self.stage = next_stage

    def assert_stage(self, required: str):
        if self.stage != required:
            raise RuntimeError(
                f"Expected stage {required}, currently at {self.stage}"
            )

    def log_llm_call(self, record: dict):
        self.llm_log.append(record)
        with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------
def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def call_llm(
    system: str,
    user: str,
    stage_label: str,
    call_id: str,
    input_artifacts: list[str],
    output_artifact: str,
    state: PipelineState,
    qa_scores_included: bool = False,
) -> str:
    full_prompt = system + "\n" + user
    p_hash = prompt_hash(full_prompt)

    api_key = os.environ.get("OPENAI_API_KEY", "")

    # OpenAI Chat Completions payload
    payload = {
        "model": OPENAI_MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Retry with exponential backoff for rate limit (429) errors
    max_retries = 5
    wait = 15  # seconds to wait on first 429
    for attempt in range(max_retries):
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=120)
        if response.status_code == 429:
            print(f"   Rate limited — waiting {wait}s before retry (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
            wait *= 2  # exponential backoff: 15 -> 30 -> 60 -> 120 -> 240
            continue
        response.raise_for_status()
        break
    else:
        raise RuntimeError(f"OpenAI API rate limit exceeded after {max_retries} retries")

    data = response.json()

    # Extract text from OpenAI response structure
    result_text = data["choices"][0]["message"]["content"]

    # Small pause between calls to avoid rate limits
    time.sleep(2)

    log_record = {
        "stage": stage_label,
        "call_id": call_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "model": OPENAI_MODEL,
        "prompt_hash": p_hash,
        "input_artifacts": input_artifacts,
        "output_artifact": output_artifact,
        "qa_scores_included": qa_scores_included,
    }
    state.log_llm_call(log_record)
    return result_text


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------
def extract_json_block(text: str) -> str:
    """Extract the first JSON block from LLM output (with or without fences)."""
    # Try fenced block first
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        return m.group(1).strip()
    # Otherwise find first { or [
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start != -1:
            # Find matching end
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == start_char:
                    depth += 1
                elif c == end_char:
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
    return text.strip()


# ---------------------------------------------------------------------------
# STAGE 1: Load inputs
# ---------------------------------------------------------------------------
def stage_load_inputs(state: PipelineState):
    state.assert_stage("INIT")
    print("\n[1] Loading inputs...")

    if not QA_FRAMEWORK_FILE.exists():
        raise FileNotFoundError(f"Missing {QA_FRAMEWORK_FILE}")
    with open(QA_FRAMEWORK_FILE, encoding="utf-8") as f:
        state.framework = json.load(f)

    txt_files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No transcript .txt files found in {TRANSCRIPTS_DIR}")
    state.transcripts = txt_files
    print(f"   Loaded framework with {len(state.framework['dimensions'])} dimensions")
    print(f"   Found {len(txt_files)} transcript files")

    # Reset log file
    LLM_LOG_FILE.write_text("")

    state.advance("INPUTS_LOADED")


# ---------------------------------------------------------------------------
# STAGE 2: Parse transcripts (deterministic, no LLM)
# ---------------------------------------------------------------------------
def parse_transcript_file(path: Path) -> dict:
    """
    Parse a transcript file into structured records.
    Header format: --- CALL_ID (Agent: NAME) ---
    Body lines: SPEAKER: text
    """
    text = path.read_text(encoding="utf-8")
    lines = [l.rstrip() for l in text.splitlines()]

    # Extract header
    call_id = path.stem
    agent_name = "Unknown"
    header_re = re.compile(r"---\s*(\S+)\s*\(Agent:\s*([^)]+)\)\s*---")
    for line in lines:
        m = header_re.match(line)
        if m:
            call_id = m.group(1).strip()
            agent_name = m.group(2).strip()
            break

    # Parse turns
    turns = []
    turn_number = 0
    for line in lines:
        if not line or header_re.match(line):
            continue
        colon_idx = line.find(":")
        if colon_idx == -1:
            continue
        speaker = line[:colon_idx].strip()
        text = line[colon_idx + 1 :].strip()
        if not text:
            continue
        turn_number += 1
        turns.append(
            {
                "call_id": call_id,
                "agent_name": agent_name,
                "turn_number": turn_number,
                "speaker": speaker,
                "text": text,
            }
        )

    # Derived metadata (deterministic)
    estimated_duration_minutes = max(1, round(len(turns) * 0.75))

    agent_turns = [t for t in turns if t["speaker"] == agent_name]
    customer_turns = [t for t in turns if t["speaker"] != agent_name]

    # Issue type heuristics
    all_customer_text = " ".join(t["text"].lower() for t in customer_turns)
    if any(w in all_customer_text for w in ["withdrawal", "deposit", "money"]):
        issue_type = "financial_transaction"
    elif any(w in all_customer_text for w in ["suspended", "banned", "blocked", "account"]):
        issue_type = "account_access"
    elif any(w in all_customer_text for w in ["leverage", "trading", "forex", "regulation"]):
        issue_type = "trading_query"
    else:
        issue_type = "general_inquiry"

    # Resolution heuristics
    last_customer = customer_turns[-1]["text"].lower() if customer_turns else ""
    resolution_status = (
        "resolved"
        if any(w in last_customer for w in ["thank", "ok", "okay", "perfect", "great", "helpful"])
        else "unresolved"
    )

    # Escalation signals
    escalation_keywords = [
        "report", "complain", "manager", "unacceptable", "terrible",
        "ridiculous", "lawsuit", "legal", "ombudsman",
    ]
    escalation_signals = [
        t["text"]
        for t in customer_turns
        if any(kw in t["text"].lower() for kw in escalation_keywords)
    ]

    return {
        "call_id": call_id,
        "agent_name": agent_name,
        "turns": turns,
        "metadata": {
            "estimated_duration_minutes": estimated_duration_minutes,
            "total_turns": len(turns),
            "agent_turns": len(agent_turns),
            "customer_turns": len(customer_turns),
            "issue_type": issue_type,
            "resolution_status": resolution_status,
            "customer_escalation_signals": escalation_signals,
        },
    }


def stage_parse_transcripts(state: PipelineState):
    state.assert_stage("INPUTS_LOADED")
    print("\n[2] Parsing transcripts...")
    PARSED_DIR.mkdir(exist_ok=True)

    for path in state.transcripts:
        parsed = parse_transcript_file(path)
        call_id = parsed["call_id"]
        state.parsed[call_id] = parsed

        out = PARSED_DIR / f"{call_id}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)
        print(f"   Parsed {call_id} ({parsed['metadata']['total_turns']} turns) -> {out.name}")

    state.advance("TRANSCRIPTS_PARSED")


# ---------------------------------------------------------------------------
# STAGE 3: QA Scoring (one LLM call per transcript)
# ---------------------------------------------------------------------------
def stage_qa_scoring(state: PipelineState):
    state.assert_stage("TRANSCRIPTS_PARSED")
    print("\n[3] QA Scoring (Stage 1 LLM calls)...")

    all_scores = {}

    for call_id, parsed in state.parsed.items():
        agent_name = parsed["agent_name"]
        print(f"   Scoring {call_id} (Agent: {agent_name})...")

        turns_text = "\n".join(
            f"[{t['turn_number']}] {t['speaker']}: {t['text']}"
            for t in parsed["turns"]
        )

        framework_text = json.dumps(state.framework, indent=2)

        system_prompt = textwrap.dedent("""
            You are a strict QA evaluator for customer support calls.
            Respond ONLY with valid JSON — no preamble, no markdown fences, no commentary.
        """).strip()

        user_prompt = textwrap.dedent(f"""
            Evaluate the following customer support transcript against the QA framework.

            ## QA Framework
            {framework_text}

            ## Transcript ({call_id}, Agent: {agent_name})
            {turns_text}

            ## Instructions
            Score EACH dimension D1-D7 on a scale of 0-10.
            For EACH auto-fail condition, state whether it was triggered.

            Return ONLY this JSON structure (no markdown, no extra text):
            {{
              "call_id": "{call_id}",
              "agent_name": "{agent_name}",
              "dimension_scores": [
                {{
                  "dimension_id": "D1",
                  "score": <0-10 integer>,
                  "evidence_quote": "<exact quote from transcript or 'N/A'>",
                  "rationale": "<one sentence>"
                }}
                // ... repeat for D2-D7
              ],
              "auto_fail_checks": [
                {{
                  "condition": "<exact condition text>",
                  "triggered": <true|false>,
                  "evidence": "<exact quote or null>"
                }}
                // ... one entry per auto-fail condition
              ]
            }}
        """).strip()

        raw = call_llm(
            system=system_prompt,
            user=user_prompt,
            stage_label="QA_SCORING",
            call_id=call_id,
            input_artifacts=[
                str(PARSED_DIR / f"{call_id}.json"),
                str(QA_FRAMEWORK_FILE),
            ],
            output_artifact=str(QA_SCORES_FILE),
            state=state,
            qa_scores_included=False,
        )

        try:
            result = json.loads(extract_json_block(raw))
        except json.JSONDecodeError as e:
            print(f"   ERROR parsing QA JSON for {call_id}: {e}")
            print(f"   Raw: {raw[:500]}")
            raise

        # Compute weighted total deterministically in code
        dim_map = {d["id"]: d["weight"] for d in state.framework["dimensions"]}
        weighted_total = 0.0
        for ds in result["dimension_scores"]:
            w = dim_map.get(ds["dimension_id"], 0)
            weighted_total += ds["score"] * w

        # Normalise to 0-100
        weighted_total_100 = weighted_total * 10

        result["weighted_total"] = round(weighted_total_100, 2)
        result["auto_fail_triggered"] = any(
            ac["triggered"] for ac in result["auto_fail_checks"]
        )

        all_scores[call_id] = result
        print(f"   {call_id}: weighted={result['weighted_total']:.1f}  auto_fail={result['auto_fail_triggered']}")

    state.qa_scores = all_scores

    with open(QA_SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2)
    print(f"   Saved -> {QA_SCORES_FILE.name}")

    state.advance("QA_SCORED")


# ---------------------------------------------------------------------------
# STAGE 4: Compliance Extraction (one LLM call per transcript)
# ---------------------------------------------------------------------------
def stage_compliance_extraction(state: PipelineState):
    state.assert_stage("QA_SCORED")
    print("\n[4] Compliance Extraction (Stage 2 LLM calls)...")

    all_flags = {}

    for call_id, parsed in state.parsed.items():
        agent_name = parsed["agent_name"]
        print(f"   Extracting compliance for {call_id}...")

        turns_text = "\n".join(
            f"[{t['turn_number']}] {t['speaker']}: {t['text']}"
            for t in parsed["turns"]
        )

        system_prompt = textwrap.dedent("""
            You are a compliance risk specialist reviewing customer support call transcripts.
            Focus ONLY on compliance, legal, financial, regulatory, reputational, and conduct risks.
            Respond ONLY with valid JSON — no preamble, no markdown fences, no commentary.
        """).strip()

        user_prompt = textwrap.dedent(f"""
            Review the following transcript for compliance and conduct risks.

            ## Transcript ({call_id}, Agent: {agent_name})
            {turns_text}

            Return a JSON array of risk flags. Each flag must have this structure:
            {{
              "call_id": "{call_id}",
              "statement_text": "<exact quote from transcript>",
              "speaker": "<speaker name>",
              "risk_type": "<regulatory | financial_commitment | data_disclosure | conduct | reputational | other>",
              "severity": "<critical | high | medium | low>",
              "explanation": "<one or two sentences explaining the risk>"
            }}

            If there are no compliance risks, return an empty array: []

            Return ONLY the JSON array, no other text.
        """).strip()

        raw = call_llm(
            system=system_prompt,
            user=user_prompt,
            stage_label="COMPLIANCE_EXTRACTION",
            call_id=call_id,
            input_artifacts=[str(PARSED_DIR / f"{call_id}.json")],
            output_artifact=str(COMPLIANCE_FLAGS_FILE),
            state=state,
            qa_scores_included=False,
        )

        try:
            flags = json.loads(extract_json_block(raw))
        except json.JSONDecodeError as e:
            print(f"   ERROR parsing compliance JSON for {call_id}: {e}")
            flags = []

        all_flags[call_id] = flags
        print(f"   {call_id}: {len(flags)} compliance flag(s)")

    state.compliance_flags = all_flags

    with open(COMPLIANCE_FLAGS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_flags, f, indent=2)
    print(f"   Saved -> {COMPLIANCE_FLAGS_FILE.name}")

    state.advance("COMPLIANCE_EXTRACTED")


# ---------------------------------------------------------------------------
# STAGE 5: Coaching Notes (one LLM call per transcript; NO QA scores)
# ---------------------------------------------------------------------------
def stage_coaching(state: PipelineState):
    state.assert_stage("COMPLIANCE_EXTRACTED")
    print("\n[5] Generating Coaching Notes (Stage 3 LLM calls — NO QA scores)...")

    coaching_parts = []

    for call_id, parsed in state.parsed.items():
        agent_name = parsed["agent_name"]
        print(f"   Coaching {call_id} (Agent: {agent_name})...")

        turns_text = "\n".join(
            f"[{t['turn_number']}] {t['speaker']}: {t['text']}"
            for t in parsed["turns"]
        )

        # ENFORCE: coaching prompt must NOT include QA scores, weighted totals,
        # dashboard grades, or compliance severity scores.
        # We construct the prompt here and verify these are absent.
        system_prompt = textwrap.dedent("""
            You are an experienced customer support coach.
            Your coaching must be grounded ONLY in evidence from the transcript.
            Do NOT reference any scores, grades, or compliance ratings.
            Respond ONLY with valid JSON — no preamble, no markdown fences, no commentary.
        """).strip()

        user_prompt = textwrap.dedent(f"""
            Provide personalised coaching notes for agent {agent_name} based solely on the transcript below.

            ## Call ID: {call_id}
            ## Agent: {agent_name}

            ## Transcript
            {turns_text}

            Return a JSON object with this exact structure:
            {{
              "call_id": "{call_id}",
              "agent_name": "{agent_name}",
              "what_went_well": [
                {{
                  "point": "<observation>",
                  "agent_quote": "<exact quote from transcript>"
                }},
                {{
                  "point": "<observation>",
                  "agent_quote": "<exact quote from transcript>"
                }}
              ],
              "development_areas": [
                {{
                  "point": "<observation>",
                  "agent_quote": "<exact quote from transcript>",
                  "suggested_alternative": "<suggested rephrasing>"
                }},
                {{
                  "point": "<observation>",
                  "agent_quote": "<exact quote from transcript>",
                  "suggested_alternative": "<suggested rephrasing>"
                }}
              ]
            }}

            Rules:
            - what_went_well must have at least 2 items
            - development_areas must have at least 2 items
            - Every item must include an exact agent_quote from the transcript
            - Return ONLY the JSON object, no other text
        """).strip()

        # Verify coaching prompt does NOT contain QA scores or grades
        _verify_no_qa_scores_in_prompt(user_prompt, system_prompt, call_id)

        raw = call_llm(
            system=system_prompt,
            user=user_prompt,
            stage_label="COACHING",
            call_id=call_id,
            input_artifacts=[str(PARSED_DIR / f"{call_id}.json")],
            output_artifact=str(COACHING_NOTES_FILE),
            state=state,
            qa_scores_included=False,  # Enforced
        )

        try:
            coaching = json.loads(extract_json_block(raw))
        except json.JSONDecodeError as e:
            print(f"   ERROR parsing coaching JSON for {call_id}: {e}")
            coaching = {"call_id": call_id, "agent_name": agent_name, "raw": raw}

        state.coaching_notes[call_id] = coaching
        coaching_parts.append(coaching)

    # Render coaching notes to markdown
    _write_coaching_markdown(coaching_parts)
    print(f"   Saved -> {COACHING_NOTES_FILE.name}")

    state.advance("COACHING_GENERATED")


def _verify_no_qa_scores_in_prompt(user_prompt: str, system_prompt: str, call_id: str):
    """Raise an error if QA scores or grade data leaked into the coaching prompt."""
    combined = (user_prompt + system_prompt).lower()
    forbidden_patterns = [
        "weighted_total", "weighted total", "auto_fail_triggered",
        r"\bgrade\b", "dimension_score", "d1.*score", "d2.*score",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, combined):
            raise RuntimeError(
                f"QA score data detected in coaching prompt for {call_id}: '{pattern}'"
            )


def _write_coaching_markdown(coaching_list: list):
    lines = ["# Agent Coaching Notes\n", f"_Generated: {datetime.now().isoformat()}_\n", "---\n"]
    for c in coaching_list:
        call_id = c.get("call_id", "?")
        agent = c.get("agent_name", "?")
        lines.append(f"\n## {call_id} — Agent: {agent}\n")

        lines.append("\n### ✅ What Went Well\n")
        for item in c.get("what_went_well", []):
            lines.append(f"**{item.get('point', '')}**\n")
            lines.append(f"> _{item.get('agent_quote', '')}_\n\n")

        lines.append("\n### 🔧 Development Areas\n")
        for item in c.get("development_areas", []):
            lines.append(f"**{item.get('point', '')}**\n")
            lines.append(f"> _{item.get('agent_quote', '')}_\n\n")
            lines.append(f"💡 **Suggested alternative:** {item.get('suggested_alternative', '')}\n\n")

        lines.append("\n---\n")

    with open(COACHING_NOTES_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# STAGE 6: Dashboard Summary (deterministic code)
# ---------------------------------------------------------------------------
def stage_dashboard(state: PipelineState):
    state.assert_stage("COACHING_GENERATED")
    print("\n[6] Computing Dashboard Summary...")

    severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}

    rows = []
    for call_id, parsed in state.parsed.items():
        qa = state.qa_scores.get(call_id, {})
        flags = state.compliance_flags.get(call_id, [])

        weighted = qa.get("weighted_total", 0.0)
        auto_fail = qa.get("auto_fail_triggered", False)
        grade = compute_grade(weighted, auto_fail)

        flag_count = len(flags)
        highest_severity = "none"
        if flags:
            best = max(flags, key=lambda f: severity_rank.get(f.get("severity", "low"), 0))
            highest_severity = best.get("severity", "none")

        rows.append(
            {
                "call_id": call_id,
                "agent_name": parsed["agent_name"],
                "weighted_qa_score": weighted,
                "auto_fail_triggered": auto_fail,
                "compliance_flags_count": flag_count,
                "highest_compliance_severity": highest_severity,
                "grade": grade,
            }
        )

    _write_dashboard_markdown(rows)
    print(f"   Saved -> {DASHBOARD_FILE.name}")

    # Store rows for later use
    state._dashboard_rows = rows

    state.advance("DASHBOARD_COMPUTED")


def _write_dashboard_markdown(rows: list):
    lines = [
        "# QA Dashboard Summary\n",
        f"_Generated: {datetime.now().isoformat()}_\n\n",
        "| Agent | Call ID | QA Score | Auto-Fail | Compliance Flags | Highest Severity | Grade |\n",
        "|-------|---------|----------|-----------|-----------------|-----------------|-------|\n",
    ]
    for r in rows:
        af = "✗ YES" if r["auto_fail_triggered"] else "✓ No"
        lines.append(
            f"| {r['agent_name']} | {r['call_id']} | {r['weighted_qa_score']:.1f} "
            f"| {af} | {r['compliance_flags_count']} "
            f"| {r['highest_compliance_severity']} | **{r['grade']}** |\n"
        )

    lines.append("\n## Grade Key\n")
    lines.append("| Grade | Threshold |\n|-------|-----------|\n")
    lines.append("| A | ≥ 85 |\n| B | ≥ 70 |\n| C | ≥ 55 |\n| D | ≥ 40 |\n| F | < 40 or auto-fail |\n")

    with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# STAGE 7: Calibration Check (SHOULD ATTEMPT)
# ---------------------------------------------------------------------------
def stage_calibration(state: PipelineState):
    print("\n[7] Calibration Check (single LLM call)...")

    score_sheets = {cid: state.qa_scores[cid] for cid in state.qa_scores}
    score_sheets_text = json.dumps(score_sheets, indent=2)

    system_prompt = textwrap.dedent("""
        You are a QA calibration reviewer. You compare score sheets across multiple calls
        to identify whether similar agent behaviours were scored inconsistently.
        Respond ONLY with valid JSON — no preamble, no markdown fences, no commentary.
    """).strip()

    user_prompt = textwrap.dedent(f"""
        Review the following QA score sheets for scoring inconsistencies across transcripts.

        ## All Score Sheets
        {score_sheets_text}

        Identify any cases where similar agent behaviours appear to have been scored
        inconsistently across different calls.

        Return a JSON array of inconsistency objects:
        [
          {{
            "dimension_id": "<D1-D7>",
            "call_ids": ["<id1>", "<id2>"],
            "observed_difference": "<describe the scoring difference>",
            "why_it_may_be_inconsistent": "<explanation>",
            "recommended_adjustment": "<suggested correction>"
          }}
        ]

        If no inconsistencies found, return: []
        Return ONLY the JSON array.
    """).strip()

    raw = call_llm(
        system=system_prompt,
        user=user_prompt,
        stage_label="CALIBRATION_CHECK",
        call_id="ALL",
        input_artifacts=[str(QA_SCORES_FILE)],
        output_artifact=str(CALIBRATION_FILE),
        state=state,
        qa_scores_included=True,
    )

    try:
        result = json.loads(extract_json_block(raw))
    except json.JSONDecodeError:
        result = []

    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"   Found {len(result)} calibration note(s) -> {CALIBRATION_FILE.name}")


# ---------------------------------------------------------------------------
# STAGE 8: Team Performance Trend (deterministic code)
# ---------------------------------------------------------------------------
def stage_team_trend(state: PipelineState):
    print("\n[8] Computing Team Performance Trend...")

    BASELINE = {"D1": 8, "D2": 7, "D3": 7, "D4": 6, "D5": 7, "D6": 7, "D7": 8}

    # Compute current averages per dimension deterministically
    dim_totals: dict[str, list[float]] = {d["id"]: [] for d in state.framework["dimensions"]}

    for call_id, qa in state.qa_scores.items():
        for ds in qa.get("dimension_scores", []):
            did = ds["dimension_id"]
            if did in dim_totals:
                dim_totals[did].append(ds["score"])

    current_averages = {
        did: round(sum(scores) / len(scores), 2) if scores else 0.0
        for did, scores in dim_totals.items()
    }

    trend = {}
    for did, current in current_averages.items():
        baseline = BASELINE.get(did, 0)
        delta = round(current - baseline, 2)
        trend[did] = {
            "dimension_id": did,
            "baseline": baseline,
            "current_average": current,
            "delta": delta,
            "direction": "up" if delta > 0 else ("down" if delta < 0 else "flat"),
        }

    with open(TEAM_TREND_FILE, "w", encoding="utf-8") as f:
        json.dump(trend, f, indent=2)
    print(f"   Saved -> {TEAM_TREND_FILE.name}")


# ---------------------------------------------------------------------------
# STAGE 9: Auto-Escalation Logic (STRETCH)
# ---------------------------------------------------------------------------
def stage_escalation(state: PipelineState):
    print("\n[9] Auto-Escalation Logic...")

    cases = []
    severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}

    for call_id, parsed in state.parsed.items():
        qa = state.qa_scores.get(call_id, {})
        flags = state.compliance_flags.get(call_id, [])
        agent_name = parsed["agent_name"]

        # Auto-fail trigger
        if qa.get("auto_fail_triggered"):
            for ac in qa.get("auto_fail_checks", []):
                if ac.get("triggered"):
                    cases.append(
                        {
                            "call_id": call_id,
                            "agent_name": agent_name,
                            "trigger_type": "auto_fail",
                            "triggered_condition": ac["condition"],
                            "transcript_excerpt": ac.get("evidence") or "See transcript",
                            "recommended_action": "Immediate supervisor review required. Do not replay recording to customer without legal approval.",
                        }
                    )

        # Critical compliance flag
        for flag in flags:
            if flag.get("severity") == "critical":
                cases.append(
                    {
                        "call_id": call_id,
                        "agent_name": agent_name,
                        "trigger_type": "critical_compliance",
                        "triggered_condition": flag.get("explanation", ""),
                        "transcript_excerpt": flag.get("statement_text", ""),
                        "recommended_action": "Escalate to Compliance team within 24 hours. Flag in agent performance record.",
                    }
                )

    with open(ESCALATION_FILE, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)
    print(f"   {len(cases)} escalation case(s) -> {ESCALATION_FILE.name}")


# ---------------------------------------------------------------------------
# STAGE 10: Rebuttal Coaching (STRETCH)
# ---------------------------------------------------------------------------
def stage_rebuttal_coaching(state: PipelineState):
    print("\n[10] Rebuttal Coaching...")

    rebuttal_calls = []
    for call_id, parsed in state.parsed.items():
        signals = parsed["metadata"].get("customer_escalation_signals", [])
        if signals:
            rebuttal_calls.append((call_id, parsed))

    if not rebuttal_calls:
        print("    No escalation signals found; skipping rebuttal coaching.")
        REBUTTAL_FILE.write_text("# Rebuttal Coaching\n\nNo rebuttal scenarios required for current transcripts.\n")
        return

    scenarios = []

    for call_id, parsed in rebuttal_calls:
        agent_name = parsed["agent_name"]
        turns_text = "\n".join(
            f"[{t['turn_number']}] {t['speaker']}: {t['text']}"
            for t in parsed["turns"]
        )

        system_prompt = textwrap.dedent("""
            You are a customer support de-escalation coach.
            Generate role-play coaching scenarios for handling upset or threatening customers.
            Respond ONLY with valid JSON — no preamble, no markdown fences, no commentary.
        """).strip()

        user_prompt = textwrap.dedent(f"""
            A customer threatened to report the agent or escalate in this call.
            Create a rebuttal coaching scenario for agent {agent_name} based on the transcript.

            ## Transcript ({call_id})
            {turns_text}

            Return a JSON object:
            {{
              "call_id": "{call_id}",
              "agent_name": "{agent_name}",
              "trigger_moment": "<exact customer quote that triggered the escalation>",
              "simulated_customer_message": "<a realistic challenging customer message>",
              "ideal_agent_response": "<the ideal de-escalation response>",
              "coaching_goal": "<what this scenario teaches>",
              "technique_demonstrated": "<de-escalation technique name and description>"
            }}

            Return ONLY the JSON object.
        """).strip()

        raw = call_llm(
            system=system_prompt,
            user=user_prompt,
            stage_label="REBUTTAL_COACHING",
            call_id=call_id,
            input_artifacts=[str(PARSED_DIR / f"{call_id}.json")],
            output_artifact=str(REBUTTAL_FILE),
            state=state,
            qa_scores_included=False,
        )

        try:
            scenario = json.loads(extract_json_block(raw))
        except json.JSONDecodeError:
            scenario = {"call_id": call_id, "raw": raw}

        scenarios.append(scenario)

    _write_rebuttal_markdown(scenarios)
    print(f"   {len(scenarios)} rebuttal scenario(s) -> {REBUTTAL_FILE.name}")


def _write_rebuttal_markdown(scenarios: list):
    lines = ["# Rebuttal Coaching Scenarios\n", f"_Generated: {datetime.now().isoformat()}_\n", "---\n"]
    for s in scenarios:
        call_id = s.get("call_id", "?")
        agent = s.get("agent_name", "?")
        lines.append(f"\n## {call_id} — Agent: {agent}\n\n")
        lines.append(f"**Trigger Moment:**\n> _{s.get('trigger_moment', '')}_\n\n")
        lines.append(f"**Coaching Goal:** {s.get('coaching_goal', '')}\n\n")
        lines.append(f"**Technique Demonstrated:** {s.get('technique_demonstrated', '')}\n\n")
        lines.append("### Role-Play Scenario\n\n")
        lines.append(f"**Simulated Customer:** {s.get('simulated_customer_message', '')}\n\n")
        lines.append(f"**Ideal Agent Response:** {s.get('ideal_agent_response', '')}\n\n")
        lines.append("---\n")

    with open(REBUTTAL_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Validation stage
# ---------------------------------------------------------------------------
def stage_validation(state: PipelineState):
    state.assert_stage("DASHBOARD_COMPUTED")
    print("\n[V] Validation...")
    _run_validation(state)
    state.advance("VALIDATION_COMPLETE")


def _run_validation(state: PipelineState = None, standalone: bool = False):
    errors = []
    warnings = []

    def check(condition: bool, msg: str):
        if not condition:
            errors.append(msg)

    def warn(condition: bool, msg: str):
        if not condition:
            warnings.append(f"WARN: {msg}")

    # Required files
    for f in [QA_FRAMEWORK_FILE, QA_SCORES_FILE, COMPLIANCE_FLAGS_FILE,
              COACHING_NOTES_FILE, DASHBOARD_FILE, LLM_LOG_FILE]:
        check(f.exists(), f"Missing required artifact: {f.name}")

    # JSON validity
    for f in [QA_FRAMEWORK_FILE, QA_SCORES_FILE, COMPLIANCE_FLAGS_FILE]:
        if f.exists():
            try:
                with open(f) as fh:
                    json.load(fh)
            except json.JSONDecodeError as e:
                check(False, f"Invalid JSON in {f.name}: {e}")

    # Transcripts were parsed
    if TRANSCRIPTS_DIR.exists():
        for txt in TRANSCRIPTS_DIR.glob("*.txt"):
            parsed_path = PARSED_DIR / f"{txt.stem}.json"
            check(parsed_path.exists(), f"Missing parsed transcript for {txt.name}")

    # LLM call log checks
    if LLM_LOG_FILE.exists():
        llm_records = []
        with open(LLM_LOG_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        llm_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        errors.append(f"Invalid JSON line in llm_calls.jsonl: {line[:80]}")

        # Per-transcript stage checks
        if QA_SCORES_FILE.exists():
            with open(QA_SCORES_FILE) as f:
                scores_data = json.load(f)
            call_ids = list(scores_data.keys())

            for cid in call_ids:
                records_for_call = [r for r in llm_records if r.get("call_id") == cid]
                stages_for_call = [r["stage"] for r in records_for_call]

                check("QA_SCORING" in stages_for_call, f"No QA_SCORING record for {cid}")
                check("COMPLIANCE_EXTRACTION" in stages_for_call, f"No COMPLIANCE_EXTRACTION record for {cid}")
                check("COACHING" in stages_for_call, f"No COACHING record for {cid}")

                for r in records_for_call:
                    if r["stage"] == "COACHING":
                        check(
                            r.get("qa_scores_included") is False,
                            f"Coaching call for {cid} has qa_scores_included != false",
                        )

        # QA framework dimension checks
        if QA_FRAMEWORK_FILE.exists() and QA_SCORES_FILE.exists():
            with open(QA_FRAMEWORK_FILE) as f:
                fw = json.load(f)
            with open(QA_SCORES_FILE) as f:
                scores = json.load(f)
            fw_dim_ids = {d["id"] for d in fw["dimensions"]}
            for cid, qa in scores.items():
                scored_dims = {ds["dimension_id"] for ds in qa.get("dimension_scores", [])}
                for did in fw_dim_ids:
                    check(did in scored_dims, f"Dimension {did} not scored in {cid}")

                # Auto-fail conditions present
                fw_conditions = set(fw.get("auto_fail_conditions", []))
                scored_conditions = {
                    ac["condition"] for ac in qa.get("auto_fail_checks", [])
                }
                for cond in fw_conditions:
                    warn(cond in scored_conditions, f"Auto-fail condition not checked in {cid}: {cond[:50]}")

        # Weighted score verification
        if QA_FRAMEWORK_FILE.exists() and QA_SCORES_FILE.exists():
            with open(QA_FRAMEWORK_FILE) as f:
                fw = json.load(f)
            with open(QA_SCORES_FILE) as f:
                scores = json.load(f)
            dim_map = {d["id"]: d["weight"] for d in fw["dimensions"]}
            for cid, qa in scores.items():
                recalc = sum(
                    ds["score"] * dim_map.get(ds["dimension_id"], 0)
                    for ds in qa.get("dimension_scores", [])
                ) * 10
                stored = qa.get("weighted_total", -1)
                check(
                    abs(recalc - stored) < 0.1,
                    f"Weighted score mismatch for {cid}: stored={stored}, recalc={recalc:.2f}",
                )

        # Dashboard rows
        if DASHBOARD_FILE.exists() and QA_SCORES_FILE.exists():
            with open(QA_SCORES_FILE) as f:
                scores = json.load(f)
            dashboard_text = DASHBOARD_FILE.read_text()
            for cid in scores:
                check(cid in dashboard_text, f"Call {cid} missing from dashboard")

    # Grade thresholds are deterministic (check that grade matches expected)
    if QA_SCORES_FILE.exists() and DASHBOARD_FILE.exists():
        with open(QA_SCORES_FILE) as f:
            scores = json.load(f)
        dash_text = DASHBOARD_FILE.read_text()
        for cid, qa in scores.items():
            expected_grade = compute_grade(
                qa.get("weighted_total", 0), qa.get("auto_fail_triggered", False)
            )
            check(
                expected_grade in dash_text,
                f"Expected grade {expected_grade} for {cid} not found in dashboard",
            )

    print(f"\n   Validation: {len(errors)} error(s), {len(warnings)} warning(s)")
    for e in errors:
        print(f"   ✗ {e}")
    for w in warnings:
        print(f"   {w}")

    if errors:
        if standalone:
            sys.exit(1)
        else:
            raise RuntimeError(f"Validation failed with {len(errors)} error(s)")
    else:
        print("   ✓ All validation checks passed")

    return errors, warnings


# ---------------------------------------------------------------------------
# Finalise
# ---------------------------------------------------------------------------
def stage_finalise(state: PipelineState):
    state.assert_stage("VALIDATION_COMPLETE")
    print("\n[F] Finalising results...")

    summary = {
        "pipeline_completed": datetime.now(timezone.utc).isoformat(),
        "transcripts_processed": len(state.parsed),
        "llm_calls_made": len(state.llm_log),
        "artifacts": [
            str(p.relative_to(ROOT))
            for p in ROOT.glob("*.json")
        ] + [
            str(p.relative_to(ROOT))
            for p in ROOT.glob("*.md")
        ] + [
            str(p.relative_to(ROOT))
            for p in ROOT.glob("*.jsonl")
        ],
    }
    with open(ROOT / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    state.advance("RESULTS_FINALISED")
    print("\n✓ Pipeline complete. Stage: RESULTS_FINALISED")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_pipeline():
    print("=" * 60)
    print("  QA PIPELINE — Starting")
    print("=" * 60)

    state = PipelineState()

    stage_load_inputs(state)
    stage_parse_transcripts(state)
    stage_qa_scoring(state)
    stage_compliance_extraction(state)
    stage_coaching(state)
    stage_dashboard(state)

    # SHOULD ATTEMPT
    stage_calibration(state)
    stage_team_trend(state)

    # STRETCH
    stage_escalation(state)
    stage_rebuttal_coaching(state)

    stage_validation(state)
    stage_finalise(state)

    print("\n" + "=" * 60)
    print("  All artifacts generated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()

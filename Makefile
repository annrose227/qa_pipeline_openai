.PHONY: run validate clean install

# Detect OS
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    RM_DIR := rmdir /s /q
    RM_FILE := del /f /q
    PYTHON := python
else
    DETECTED_OS := $(shell uname -s)
    RM_DIR := rm -rf
    RM_FILE := rm -f
    PYTHON := python3
endif

install:
	$(PYTHON) -m pip install requests --quiet

run: install
	$(PYTHON) pipeline_openai.py

validate:
	$(PYTHON) validate.py

clean:
ifeq ($(DETECTED_OS),Windows)
	-$(RM_DIR) parsed_transcripts
	-$(RM_FILE) qa_scores.json compliance_flags.json coaching_notes.md dashboard_summary.md
	-$(RM_FILE) calibration_check.json team_trend.json escalation_cases.json rebuttal_coaching.md
	-$(RM_FILE) llm_calls.jsonl pipeline_summary.json
else
	$(RM_DIR) parsed_transcripts/
	$(RM_FILE) qa_scores.json compliance_flags.json coaching_notes.md dashboard_summary.md
	$(RM_FILE) calibration_check.json team_trend.json escalation_cases.json rebuttal_coaching.md
	$(RM_FILE) llm_calls.jsonl pipeline_summary.json
endif

clean-run: clean run

.DEFAULT_GOAL := run

# Automation of AI Project Evaluation 🧪🤖
### Automated Software Testing, Quality Assurance & Functional Evaluation Framework for Python AI/ML Applications

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![Testing](https://img.shields.io/badge/Testing-Automated%20CI%2FCD%20Harness-green.svg)](#)

---

## 📌 Overview

**Automation of AI Project Evaluation** is an automated software testing and quality assurance tool designed specifically for inspecting, verifying, executing, and benchmarking lightweight Python AI/ML projects (e.g. text classifiers, sentiment analyzers, NLP pipelines, chatbots, recommender engines, computer vision prototypes).

Standard unit testing frameworks (such as `pytest` or `unittest`) often fall short when evaluating unfamiliar AI submissions or grading workshop deliverables where project architectures, input/output contracts, and dependencies vary. This tool acts as an **autonomous software testing judge** that ingests raw project submissions (via GitHub URL or ZIP archive), inspects the repository, validates dependencies, statically inspects pipeline stages, runs controlled execution with resource guardrails, performs black-box functional testing, profiles performance, evaluates code quality, and produces deterministic scores and detailed audit reports.

---

## 🏗️ Architecture & Testing Pipeline

```
  ┌────────────────────────────────────────────────────────┐
  │         Student / Developer Project Submission         │
  │            (Public GitHub URL or ZIP Archive)          │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ [Stage 1] Submission Handler & Environment Setup       │
  │ • URL validation & shallow clone / safe ZIP extraction │
  │ • Isolated ephemeral workspace creation                │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ [Stage 2] Specification & Test Contract Extraction     │
  │ • Deterministic README parsing & explicit doc matching │
  │ • Extract goal, run command, input/output contract     │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ [Stage 3] Static Repository & Security Inspection      │
  │ • AST-level import discovery & syntax validation       │
  │ • Secret detection (API keys, tokens, credentials)     │
  │ • Model weight & dataset artifact sizing checks        │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ [Stage 4] Dependency & Environment Verification        │
  │ • Declared vs imported package reconciliation          │
  │ • Missing package detection & version sanity checking  │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ [Stage 5] AI Pipeline Integrity Testing                │
  │ • Static analysis: Input ➔ Preprocessing ➔ Inference   │
  │ • Consistency cross-checking: Docs vs actual AST code  │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ [Stage 6] Sandboxed Controlled Execution Engine        │
  │ • Subprocess execution (shell=False, timeout=60s)       │
  │ • Safe environment sanitization (no host API leaks)    │
  │ • Process tree monitoring & runaway task termination   │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ [Stage 7] Black-Box Functional Test Harness            │
  │ • stdin/stdout test injection                          │
  │ • Multi-strategy assertions (Exact, JSON, Numeric, Tol)│
  │ • Test provenance tracking (student, README, candidate)│
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ [Stage 8] Runtime Profiling & Performance Benchmarking │
  │ • Total runtime & startup latency measurement          │
  │ • Real-time peak memory (RSS) profiling via psutil     │
  │ • Static loop & resource inefficiency warnings         │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ [Stage 9] Static Code Quality & Maintainability Audit  │
  │ • Cyclomatic complexity & long function heuristics     │
  │ • Unused imports, parameter count, modularity checks   │
  │ • Selective qualitative AI synthesis via Gemini API    │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ [Stage 10] Deterministic Scoring & Report Generation   │
  │ • 100-point rubric calculation across 8 dimensions    │
  │ • JSON test artifact + Human-readable HTML test report │
  └────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Testing Dimensions & Rubric (100 Points)

Unlike naive LLM wrappers that fabricate scores, this framework uses **deterministic software test results** as ground truth:

| Test Dimension | Weight | Verification Method |
|---|---|---|
| **Functionality & Test Results** | **30 pts** | Subprocess exit status (0 vs error) + Automated functional test case pass rate |
| **Pipeline Correctness** | **20 pts** | AST static structural verification of Input, Preprocessing, Inference, and Output |
| **Performance & Efficiency** | **15 pts** | Subprocess runtime latency profiling + Peak memory (RSS) consumption |
| **Code Quality & Modularity** | **15 pts** | AST cyclomatic complexity, function length, unused imports, parameter counts |
| **Error Handling & Robustness** | **5 pts** | Exception handlers (`try/except/raise`), graceful crash containment |
| **Dependencies & Environment** | **5 pts** | `requirements.txt` / declared package verification vs AST detected imports |
| **Documentation & Testability** | **5 pts** | Explicit run commands, contract specifications, sample test inputs/outputs |
| **AI Project Completeness** | **5 pts** | Legitimate AI/ML library usage and model inference logic (not stub code) |
| **TOTAL** | **100 pts** | Fully traceable, reproducible, non-hallucinated score |

---

## 🚀 Key Features

* **Universal Test Ingestion**: Supports both public GitHub repository URLs and direct ZIP archives without requiring students to use a rigid folder structure.
* **Non-Destructive Testing**: Never installs foreign student packages into the evaluator's root environment.
* **Controlled Subprocess Sandbox**: Runs code using `shell=False`, isolated ephemeral workspaces, strict execution timeouts (60s default), and process tree termination to prevent fork bombs or deadlocks.
* **Multi-Format Assertion Engine**: Compares test outputs using normalized text comparison, JSON/structured equality, and floating-point numeric tolerance.
* **Graceful Failure Tolerance**: If a student's code fails to run (e.g. missing package or syntax error), the tester **does not crash** — it logs the exact stack trace, reduces the functionality score, and continues with static analysis, code quality checks, and documentation audits.
* **Selective AI Diagnostics**: Uses the official `google-genai` SDK solely for qualitative interpretation, error diagnostics, and candidate test generation — never for numeric scoring.
* **Rich Test Artifacts**: Automatically produces machine-readable JSON results (`results/raw/`), category score breakdowns (`results/scores/`), and styled HTML reports (`results/reports/`).

---

## 💻 Installation & Setup

### 1. Prerequisites
- Python 3.10+
- Git installed on your system

### 2. Clone the Repository
```bash
git clone https://github.com/Nitin56723/Automation-of-AI-project-evaluation.git
cd Automation-of-AI-project-evaluation
```

### 3. Create a Virtual Environment & Install Dependencies
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Configure API Keys (Optional)
If you wish to enable AI-assisted diagnostics and feedback, provide your Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(If omitted, the evaluator operates in 100% deterministic offline mode).*

---

## 🖥️ Running the Application

Launch the testing dashboard:
```bash
python app/main.py
```
Open your browser at:
```
http://localhost:7860
```

---

## 🧪 Running Automated Smoke Tests

To verify the evaluation engine against local test fixtures:
```bash
python tests/smoke_test.py
```

---

## 📊 Sample Test Report

When an evaluation finishes, comprehensive audit logs and reports are generated:

```
==================================================
  AUTOMATED AI TESTING AUDIT REPORT
==================================================
  Target:            SimpleAISentimentAnalysis
  Overall Score:     62.0 / 100
  Execution Status:  FAILED (dependency_error)
  Error Reason:      ModuleNotFoundError: No module named 'transformers'
--------------------------------------------------
  Functionality & Tests:      0.0 / 30  [Measured]
  Pipeline Correctness:      20.0 / 20  [Static AST]
  Performance & Efficiency:  12.0 / 15  [Static Profile]
  Code Quality:              15.0 / 15  [Static AST]
  Error Handling:             3.0 / 5   [Static AST]
  Dependencies:               3.0 / 5   [Dependency Check]
  Documentation:              5.0 / 5   [README Parsed]
  AI Completeness:            4.0 / 5   [Model Architecture]
--------------------------------------------------
  Strengths:
    ✓ End-to-end pipeline structure verified
    ✓ Requirements file present
    ✓ Clean modular design and zero hardcoded credentials
  Issues Found:
    ⚠ Execution failed: missing dependency 'transformers'
  Recommendations:
    → Ensure all third-party imports are listed in requirements.txt
==================================================
```

---

## 📂 Project Structure

```
├── app/
│   ├── __init__.py
│   ├── main.py                   # Orchestration pipeline & Gradio testing UI
│   ├── models.py                 # Dataclasses & evaluation state contracts
│   ├── submission_handler.py     # GitHub clone & secure ZIP extractor
│   ├── readme_parser.py          # Deterministic specification parser
│   ├── repository_inspector.py   # AST repository analyzer & secret scanner
│   ├── dependency_checker.py     # Package import & dependency verifier
│   ├── pipeline_analyzer.py      # AI/ML pipeline stage analyzer
│   ├── execution_engine.py       # Controlled subprocess runner & resource monitor
│   ├── functional_tester.py      # Black-box assertion test harness
│   ├── performance_analyzer.py   # Latency & memory consumption profiler
│   ├── code_quality.py           # Cyclomatic complexity & code style auditor
│   ├── llm_service.py            # Selective Gemini diagnostics & test generator
│   ├── scoring_engine.py         # 100-point deterministic evaluation engine
│   └── report_generator.py       # JSON & HTML report generator
├── demo/
│   └── sample_ai_project/        # Reference AI testing fixture
├── results/
│   ├── raw/                      # Machine-readable evaluation JSON dumps
│   ├── scores/                   # Category score breakdown JSON
│   └── reports/                  # Styled HTML testing reports
├── tests/
│   └── smoke_test.py             # Integration testing suite
├── requirements.txt              # Pinned dependencies
├── .env.example                  # Environment configuration template
└── README.md                     # Software testing tool documentation
```

---

## 🛡️ Security & Sandbox Scope

This framework is built for evaluating workshop and educational AI project submissions. It executes arbitrary Python code using subprocess boundaries, temporary workspaces, and execution timeouts. It is **not** an operating-system kernel-level hypervisor sandbox. For evaluating untrusted hostile code in multi-tenant cloud production, containerized isolation (e.g., Docker / gVisor) is recommended.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

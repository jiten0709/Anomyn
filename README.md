<h1 align="center">✨ Anomyn ✨</h1>

<p align="center">
  <strong>AI-Powered Operational Data Validation for Regulatory Reporting</strong>
</p>

> ⚠️ **Project Status: In Progress**  
> Anomyn is currently under active development. Core validation workflows are implemented, and additional enhancements (scalability, ML optimization, monitoring, production hardening) are ongoing.

---

## 🚀 Overview

**Anomyn** is an AI-driven validation system designed to ensure operational data integrity _before_ regulatory submission.

It combines:

- Deterministic rule-based validation
- Machine learning anomaly detection
- Human-in-the-loop schema confirmation

The system automatically profiles datasets, generates validation rules, allows human review, and validates new incoming datasets against confirmed schema definitions.

This project demonstrates production-minded architecture, ML + rules hybrid validation, and schema version control principles suitable for regulated environments (finance, healthcare, compliance reporting).

---

## 🎯 Core Objectives

- Prevent regulatory reporting errors
- Detect structural and behavioral anomalies
- Provide explainable validation logic
- Enable human oversight before enforcement

---

## 🧠 Key Features

### 1. Dataset Profiling

- Automatically infers:
  - Data types
  - Numeric ranges (min/max, quantiles)
  - Null-rate thresholds
  - Allowed categorical values
- Generates deterministic validation rules based on statistical profiling.

---

### 2. Unified Schema Control

- Structural schema + validation rules stored in a **single versioned artifact**
- Atomic persistence
- Local caching for concurrency-safe updates

---

### 3. Human-in-the-Loop Workflow

- Exposes auto-inferred schema to operators
- Allows:
  - Threshold adjustment
  - Rule modification
  - Approval or rejection
- Prevents premature automation risk

---

### 4. Multi-Layer Validation Engine

Incoming datasets are validated via:

1. **Deterministic Rules**
   - Regex checks
   - Numeric bounds
   - Allowed values
   - Null-rate enforcement

2. **ML Anomaly Detection (Optional Layer)**
   - Behavioral anomaly detection
   - Cross-field inconsistencies
   - Out-of-pattern records

---

## 🏗 System Architecture

| Layer           | Technology                        |
| --------------- | --------------------------------- |
| API Framework   | FastAPI                           |
| Server          | Uvicorn                           |
| Data Processing | Python, Pandas                    |
| ML Layer        | Custom anomaly detection module   |
| Persistence     | JSON schema artifacts (versioned) |

---

## 🚀 Quick Start

### Installation

1.  **Clone the repository**

    ```bash
    git clone https://github.com/jiten0709/Anomyn.git
    cd Anomyn
    ```

2.  **Set up environment**

    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```

3.  **Configure `.env`**

    ```bash
    cp .env.example .env
    ```

4.  **Start Development Server**
    ```bash
     python3 main.py
    ```

<p align="center"><em>Made with ❤️ by Jiten</em></p>

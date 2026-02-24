# Judgement OSS -- Prompt Injection Attack Console

> **DISCLAIMER:** This tool is intended for authorized security testing and educational purposes only. Only test systems you own or have explicit written permission to test. Unauthorized access to computer systems is illegal under the Computer Fraud and Abuse Act (CFAA) and equivalent laws worldwide. The authors assume no liability for misuse of this tool.

The open-source version of [Judgement](https://judgement.fallenangelsystems.com) by Fallen Angel Systems.

Test AI chatbots, APIs, and agents for prompt injection vulnerabilities. Includes an education tab for beginners and a full attack console for security professionals.

## Quick Start

### Install from PyPI
```bash
pip install fas-judgement
judgement
```

### Or run from source
```bash
git clone https://github.com/fallen-angel-systems/fas-judgement-oss.git
cd fas-judgement-oss
pip install -r requirements.txt
python -m judgement.server
```

Open `http://localhost:8668` in your browser.

### Options
```bash
judgement --port 9000        # Custom port
judgement --host 127.0.0.1   # Localhost only
```

## Features

- **Attack Console** -- Configure targets, import cURL commands, fire pattern-based attacks with live streaming results
- **Education Tab** -- Learn what prompt injection is, how to find endpoints, and how to interpret results
- **Pattern Browser** -- Search and explore attack patterns with expandable explanations
- **LLM Verdict** -- Optional Ollama integration for AI-powered response classification
- **SQLite History** -- All sessions and results stored locally
- **SSRF Protection** -- Target URL validation prevents internal network access
- **Single-Page App** -- Zero CDN dependencies, dark theme, mobile responsive

## Patterns

Place your patterns in `patterns.json` in the project root. Each pattern should have:

```json
{
  "id": "unique-id",
  "category": "jailbreak",
  "text": "The attack payload...",
  "explanation": "What this pattern does",
  "why_it_works": "Why this technique is effective",
  "difficulty": "beginner"
}
```

## Configuration

- Default port: `8668`
- Ollama URL: `OLLAMA_URL` env var (default: `http://localhost:11434`)
- Ollama model: `OLLAMA_MODEL` env var (default: `qwen2.5:14b`)

---

Want the full experience? 240K+ training data powering thousands of curated attack patterns, with weekly and monthly updates. Plus leaderboard, campaigns, and premium features at [judgement.fallenangelsystems.com](https://judgement.fallenangelsystems.com)

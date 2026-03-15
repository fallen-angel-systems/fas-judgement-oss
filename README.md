<div align="center">

# FAS Judgement

### Prompt Injection Attack Console

**Test your AI's defenses before someone else does.**

[![PyPI Version](https://img.shields.io/pypi/v/fas-judgement?color=blue)](https://pypi.org/project/fas-judgement/)
[![Downloads](https://img.shields.io/pypi/dm/fas-judgement?color=green)](https://pypi.org/project/fas-judgement/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/fallen-angel-systems/fas-judgement-oss?style=social&v=2)](https://github.com/fallen-angel-systems/fas-judgement-oss)

[Install](#quick-start) | [Demo Target](#demo-target) | [Features](#features) | [Elite](#free-vs-elite) | [Contributing](#contributing)

</div>

---

![Judgement - Shall We Play a Game?](docs/images/judgement-hero.png)

## Why Judgement?

Your AI chatbot, API, or agent is probably vulnerable to prompt injection. Most are. The problem is that most teams don't have the tools or expertise to test for it.

Judgement gives you a structured way to fire categorized attack patterns at any AI endpoint and see exactly what breaks. No security background required -- the built-in education tab teaches you as you go.

Built by [Fallen Angel Systems](https://fallenangelsystems.com), the team behind [Guardian](https://fallenangelsystems.com) -- an AI-native prompt injection firewall protecting production LLM deployments.

## What's New in v2.1.0

**Architecture overhaul** -- Judgement has been restructured from a single-file monolith into a modular DDD (Domain-Driven Design) architecture with 52 Python files across 7 layers. This makes it extensible, testable, and ready for future security modules.

### New Features
- **Demo Target** -- Built-in vulnerable AI chatbot for practice. Run `judgement demo` and fire attacks at it without needing a real AI endpoint
- **Multi-Turn Attack Engine** -- Chain attacks across multiple conversation turns with phase-aware scoring and session persistence
- **Transport Layer** -- Attack targets via HTTP, Ollama, Discord, Telegram, Slack, or headless browser
- **Module Registry** -- Pluggable security module system. AI Security is module one; future modules (Web Security, API Security, etc.) drop in without restructuring
- **Professional Reports** -- Generate HTML, Markdown, JSON, and SARIF reports with CWE/OWASP references

### Improvements
- DDD architecture: core (models, enums, errors, interfaces) / modules / transport / http / utils
- Scanner scorer with keyword heuristics + optional LLM classification
- SSRF protection, input sanitization, and cURL parser in dedicated utils
- Phase-aware multi-turn scoring with data leak detection (19 regex patterns)
- Persistent multi-turn sessions (SQLite, survive restarts)

## Quick Start

### Install from PyPI (recommended)

```bash
pip install fas-judgement
judgement
```

That's it. Open `http://localhost:8668` and start testing.

### Or run from source

```bash
git clone https://github.com/fallen-angel-systems/fas-judgement-oss.git
cd fas-judgement-oss
pip install -r requirements.txt
python -m fas_judgement
```

### CLI Commands

```bash
judgement                    # Start the scanner (port 8668)
judgement demo               # Start demo target (port 8667, default persona)
judgement demo hardened       # Demo with hardened persona (~90% block rate)
judgement demo vulnerable     # Demo with vulnerable persona (~10% block rate)
judgement activate FAS-XXXX   # Activate Elite license
judgement status              # Check license tier and pattern count
judgement deactivate          # Revert to free tier
```

### Options

```bash
judgement --port 9000        # Custom port
judgement --host 127.0.0.1   # Localhost only
judgement --host 0.0.0.0     # Expose to network
```

## Demo Target

New to prompt injection? Start here. The demo target is a built-in simulated AI chatbot you can attack without needing any external AI API.

```bash
# Terminal 1: Start the demo target
judgement demo

# Terminal 2: Start the scanner
judgement
```

Point the scanner at `http://localhost:8667/demo/chat` and fire away.

### Three Personas

| Persona | Block Rate | What It Simulates |
|---------|-----------|-------------------|
| **hardened** | ~90% | Well-tuned safety layer. Blocks injections, DAN, role-play, emotional manipulation, token smuggling |
| **default** | ~55% | Typical GPT-style deployment. Blocks obvious attacks, leaks secrets under social engineering |
| **vulnerable** | ~10% | Raw model with no guardrails. Dumps API keys, passwords, system prompt on request |

Switch personas at runtime:
```bash
curl -X POST http://localhost:8667/demo/persona -d '{"persona": "vulnerable"}'
```

## Features

### Attack Console
Configure your target (URL, headers, body template), import directly from cURL commands, and fire pattern-based attacks with **live streaming results**. Use quick presets to structure your approach:

| Preset | What It Does |
|--------|-------------|
| Smoke Test | ~15 patterns, critical+high severity, 1 per category |
| Full Sweep | ~50 patterns, proportional spread across all categories |
| Deep Dive | ~100 patterns, heavy coverage, min 2 per category |
| Critical Only | All critical+high severity patterns, no limits |

### Multi-Turn Attack Engine (Elite)
Chain attacks across multiple conversation turns. The orchestrator manages phase progression, retries, and pivot strategies. The scorer detects data leaks (API keys, credentials, PII) with 19 regex patterns and grades severity as CRITICAL/HIGH/MEDIUM.

Supports all transport types -- attack chatbots on Discord, Telegram, Slack, or any HTTP API.

### Scan Target Auto-Detect
Point Judgement at any URL and it auto-detects:
- HTTP method (POST, GET, PUT, PATCH)
- Payload field name (message, prompt, input, query, etc.)
- Required headers and auth format
- AI provider (OpenAI, Anthropic, custom)

### Professional Reports (Elite)
Generate security assessment reports from any attack session:

| Format | Use Case |
|--------|----------|
| **HTML** | Print-ready professional report with executive summary, CWE/OWASP references |
| **Markdown** | Bug bounty submissions for HackerOne, Bugcrowd, GitHub Issues |
| **JSON** | Structured data for custom tooling and dashboards |
| **SARIF** | Upload to GitHub Code Scanning or Azure DevOps |

### LLM Verdict (Optional)
Connect a local [Ollama](https://ollama.ai) instance for AI-powered response classification. More accurate than keyword matching for detecting subtle bypasses.

### Pattern Submissions
Found a novel attack technique? Submit it directly from the app. If it scores 70%+ confidence and isn't a duplicate, it gets added to the community library.

### Built-in Safety
- **SSRF Protection** -- Target URL validation prevents scanning internal/private networks
- **Local-only by default** -- Binds to localhost, no accidental exposure
- **Zero telemetry** -- Nothing phones home, ever
- **Credit protection** -- Configurable pattern limits and auto-stop on consecutive errors

## Architecture (v2.1.0)

```
fas_judgement/
├── config.py              # Environment and app configuration
├── core/                  # Domain models, enums, errors, interfaces, registry
├── modules/
│   └── ai_security/       # AI Security module (pluggable)
│       ├── scanner/       # Single-shot attack engine
│       ├── multi_turn/    # Multi-turn attack orchestrator
│       ├── patterns/      # Pattern loading, filtering, repository
│       └── demo/          # Built-in vulnerable chatbot
├── transport/             # HTTP, Ollama, Discord, Telegram, Slack, Website
├── http/                  # FastAPI app, routers, dependencies
├── ui/                    # Frontend SPA
└── utils/                 # License client, security, email, Ollama helpers
```

Future security modules (Web Security, API Security, Network, etc.) plug into `modules/` without restructuring the app.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `--port` | `8668` | Server port |
| `--host` | `127.0.0.1` | Bind address |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:14b` | Model for LLM verdict |

## Free vs Elite

| Feature | Free | Elite |
|---------|:----:|:-----:|
| Attack console with presets | Yes | Yes |
| Demo target (3 personas) | Yes | Yes |
| Severity filter and search | Yes | Yes |
| Education tab | Yes | Yes |
| LLM verdict (Ollama) | Yes | Yes |
| Scan Target auto-detect | Yes | Yes |
| Pattern submissions | Yes | Yes |
| Built-in documentation | Yes | Yes |
| Starter patterns | 100 | 34,838+ |
| Multi-turn attack chains | -- | Yes |
| Professional reports (HTML/MD/JSON/SARIF) | Basic MD | Full suite |
| Per-category attack limits | -- | Yes |
| Transport layer (Discord, Slack, etc.) | HTTP only | All |
| Phase-aware scoring + data leak detection | -- | Yes |
| Campaigns | -- | Coming Soon |

**[Get Elite Access](https://fallenangelsystems.com)**

## Contributing

Contributions are welcome! Here's how to help:

- **Bug reports** -- [Open an issue](https://github.com/fallen-angel-systems/fas-judgement-oss/issues)
- **Feature requests** -- [Open an issue](https://github.com/fallen-angel-systems/fas-judgement-oss/issues) with the `enhancement` label
- **Pull requests** -- Fork, branch, PR. Keep changes focused and include a description.
- **Pattern submissions** -- Use the Submit Pattern tab in the app to contribute directly

## Related Projects

- **[Guardian](https://fallenangelsystems.com)** -- AI-native prompt injection firewall (defense)
- **[Guardian Shield](https://github.com/jtil4201/Openclaw-Guardian-Shield)** -- Free local prompt injection scanner (OpenClaw skill)

## License

MIT -- see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by [Fallen Angel Systems](https://fallenangelsystems.com)

*If Judgement found a vulnerability in your AI, imagine what an attacker would find.*

</div>

> **DISCLAIMER:** This tool is intended for authorized security testing and educational purposes only. Only test systems you own or have explicit written permission to test. Unauthorized access to computer systems is illegal under the Computer Fraud and Abuse Act (CFAA) and equivalent laws worldwide. The authors assume no liability for misuse of this tool.

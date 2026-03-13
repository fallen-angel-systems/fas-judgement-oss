<div align="center">

# Judgement OSS

### Prompt Injection Attack Console

**Test your AI's defenses before someone else does.**

[![PyPI Version](https://img.shields.io/pypi/v/fas-judgement?color=blue)](https://pypi.org/project/fas-judgement/)
[![Downloads](https://img.shields.io/pypi/dm/fas-judgement?color=green)](https://pypi.org/project/fas-judgement/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/fallen-angel-systems/fas-judgement-oss?style=social&v=2)](https://github.com/fallen-angel-systems/fas-judgement-oss)

[Live Demo](https://judgement.fallenangelsystems.com) | [Documentation](#features) | [Install](#quick-start) | [Contributing](#contributing)

</div>

---

![Judgement - Shall We Play a Game?](docs/images/judgement-hero.png)

## Why Judgement?

Your AI chatbot, API, or agent is probably vulnerable to prompt injection. Most are. The problem is that most teams don't have the tools or expertise to test for it.

Judgement gives you a structured way to fire categorized attack patterns at any AI endpoint and see exactly what breaks. No security background required -- the built-in education tab teaches you as you go.

Built by [Fallen Angel Systems](https://fallenangelsystems.com), the team behind [Guardian](https://fallenangelsystems.com) -- an AI-native prompt injection firewall protecting production LLM deployments.

## What's New in v2.0.0

- **Attack Presets** -- Smoke Test, Full Sweep, Deep Dive, and Critical Only modes for structured testing
- **Severity Filter & Search** -- Filter patterns by severity level with a real-time search bar
- **Per-Category Limits** -- Control exactly how many patterns fire per category with pool count indicators
- **Custom Patterns** -- Build, edit, import, and export your own private pattern library (stored locally)
- **Professional Reports** -- Generate HTML, Markdown, JSON, and SARIF reports with CWE/OWASP references
- **Scan Target** -- Auto-detect API format, method, headers, and payload field with one click
- **License System** -- Activate Elite tier for 34,838+ patterns directly from the CLI
- **Full Documentation** -- Built-in Docs page with Volt's Red Team Playbook and complete feature reference
- **Pattern Submissions** -- Submit novel attack patterns to the community library

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
python -m judgement.server
```

### Options

```bash
judgement --port 9000        # Custom port
judgement --host 127.0.0.1   # Localhost only
judgement --host 0.0.0.0     # Expose to network
```

### Elite License Activation

```bash
judgement activate FAS-XXXX-XXXX-XXXX-XXXX   # Activate Elite license
judgement status                               # Check tier and pattern count
judgement deactivate                            # Revert to free tier
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

![Attack Console](docs/images/attack-console.png)

### Severity Filter & Search
Filter the pattern library by severity (Critical, High, Medium, Low) or use the Critical+High combo for focused testing. Search patterns in real-time to find exactly what you need.

### Custom Patterns
Build your own private attack library in the **My Patterns** tab:
- Add, edit, and delete patterns with category and notes
- Import/export as JSON for backup and sharing
- Include custom patterns in attacks alongside the curated library
- Stored locally in your browser -- never touches any server
- Up to 500 patterns, 10,000 characters each

### Professional Reports (Elite)
Generate security assessment reports from any attack session:

| Format | Use Case |
|--------|----------|
| **HTML** | Print-ready professional report with executive summary, CWE/OWASP references, and remediation advice |
| **Markdown** | Bug bounty submissions for HackerOne, Bugcrowd, GitHub Issues, or Jira |
| **JSON** | Structured data for custom tooling, dashboards, or API consumers |
| **SARIF** | Upload to GitHub Code Scanning, Azure DevOps, or any SARIF-compatible security dashboard |

Reports include risk ratings, detailed findings with evidence, and prioritized remediation recommendations.

### Pattern Browser
Browse, search, and explore attack patterns organized by category in a sortable table view. Each pattern shows ID, category, payload text, and severity level.

### Education Tab
New to prompt injection? The built-in education tab covers:
- What prompt injection is and why it matters
- How to find testable AI endpoints
- How to interpret scan results
- Common vulnerability categories explained

**No prior security experience needed.** The onboarding walkthrough guides you from zero to your first scan.

### Documentation
Built-in Docs page with expandable reference sections:
- **Red Team Playbook** by Volt -- structured methodology for professional AI red teaming
- Getting Started guides for API endpoints and web chatbots
- Attack Console reference with preset explanations
- Pattern categories and tier breakdown
- Verdict classification guide
- Credit protection and MCP integration docs
- Legal and ethics guidelines
- FAQ

### Scan Target
Point Judgement at any URL and click **Scan**. It auto-detects:
- HTTP method (POST, GET, PUT, PATCH)
- Payload field name (message, prompt, input, query, etc.)
- Required headers and auth format
- Response format and streaming support

### LLM Verdict (Optional)
Connect a local [Ollama](https://ollama.ai) instance to get AI-powered classification of responses. More accurate than keyword matching for detecting subtle bypasses where the target complies but wraps it in disclaimers.

### Pattern Submissions
Found a novel attack technique? Submit it directly from the **Submit Pattern** tab. Guardian AI auto-verifies your submission -- if it scores 70%+ confidence and isn't a duplicate, it gets added to the community library.

### Session History
All scan sessions and results are stored locally in SQLite. Review past scans, compare results across targets, and track your testing progress.

### Built-in Safety
- **SSRF Protection** -- Target URL validation prevents scanning internal/private networks
- **Local-only by default** -- Binds to localhost, no accidental exposure
- **Zero telemetry** -- Nothing phones home, ever
- **Auth confirmation** -- Warns before firing at authenticated endpoints
- **Credit protection** -- Configurable pattern limits and auto-stop on consecutive errors

## How It Works

```
+--------------+     +---------------+     +--------------+
|   You pick   |---->|  Judgement     |---->|  Your AI     |
|   patterns   |     |  fires them   |     |  endpoint    |
+--------------+     +-------+-------+     +-------+------+
                             |                     |
                      +------v-------+     +-------v------+
                      |  Results     |<----|  Response    |
                      |  + Verdict   |     |  captured    |
                      +--------------+     +--------------+
```

1. **Configure** -- Point Judgement at your AI endpoint (URL + headers + body template)
2. **Select** -- Choose attack presets or pick categories manually with severity filters
3. **Fire** -- Watch results stream in real-time via SSE
4. **Analyze** -- Review responses, optional LLM verdict classifies each result
5. **Report** -- Export findings as HTML, Markdown, JSON, or SARIF
6. **Fix** -- Use the findings to harden your AI's defenses

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
| Severity filter and search | Yes | Yes |
| Education tab | Yes | Yes |
| Pattern browser | Yes | Yes |
| LLM verdict (Ollama) | Yes | Yes |
| Scan Target auto-detect | Yes | Yes |
| MCP server integration | Yes | Yes |
| Built-in documentation | Yes | Yes |
| Pattern submissions | Yes | Yes |
| Starter patterns | 100 | 34,838+ |
| Custom patterns library | -- | Yes |
| Professional reports (HTML/MD/JSON/SARIF) | Basic MD | Full suite |
| Per-category attack limits | -- | Yes |
| Campaigns | -- | Coming Soon |
| Multi-turn attack chains | -- | Coming Soon |
| Credit protection controls | Yes | Yes |

**[Get Elite Access](https://fallenangelsystems.com)**

## Contributing

Contributions are welcome! Here's how to help:

- **Bug reports** -- [Open an issue](https://github.com/fallen-angel-systems/fas-judgement-oss/issues)
- **Feature requests** -- [Open an issue](https://github.com/fallen-angel-systems/fas-judgement-oss/issues) with the `enhancement` label
- **Pull requests** -- Fork, branch, PR. Keep changes focused and include a description.
- **Pattern submissions** -- Use the Submit Pattern tab in the app to contribute directly

## Related Projects

- **[Guardian](https://fallenangelsystems.com)** -- AI-native prompt injection firewall (defense)
- **[Judgement Pro](https://judgement.fallenangelsystems.com)** -- Full-featured hosted version with all Elite features

## License

MIT -- see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by [Fallen Angel Systems](https://fallenangelsystems.com)

*If Judgement found a vulnerability in your AI, imagine what an attacker would find.*

</div>

> **DISCLAIMER:** This tool is intended for authorized security testing and educational purposes only. Only test systems you own or have explicit written permission to test. Unauthorized access to computer systems is illegal under the Computer Fraud and Abuse Act (CFAA) and equivalent laws worldwide. The authors assume no liability for misuse of this tool.

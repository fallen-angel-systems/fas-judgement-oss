<div align="center">

# ⚖️ Judgement OSS

### Prompt Injection Attack Console

**Test your AI's defenses before someone else does.**

[![PyPI Version](https://img.shields.io/pypi/v/fas-judgement?color=blue)](https://pypi.org/project/fas-judgement/)
[![Downloads](https://img.shields.io/pypi/dm/fas-judgement?color=green)](https://pypi.org/project/fas-judgement/)
[![License](https://img.shields.io/github/license/fallen-angel-systems/fas-judgement-oss)](LICENSE)
[![Stars](https://img.shields.io/github/stars/fallen-angel-systems/fas-judgement-oss?style=social)](https://github.com/fallen-angel-systems/fas-judgement-oss)

[Live Demo](https://judgement.fallenangelsystems.com) · [Documentation](#features) · [Install](#quick-start) · [Contributing](#contributing)

</div>

---

![Judgement — Shall We Play a Game?](docs/images/judgement-hero.png)

## Why Judgement?

Your AI chatbot, API, or agent is probably vulnerable to prompt injection. Most are. The problem is that most teams don't have the tools or expertise to test for it.

Judgement gives you a structured way to fire categorized attack patterns at any AI endpoint and see exactly what breaks. No security background required — the built-in education tab teaches you as you go.

Built by [Fallen Angel Systems](https://fallenangelsystems.com), the team behind [Guardian](https://fallenangelsystems.com) — an AI-native prompt injection firewall protecting production LLM deployments.

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

## Features

### 🎯 Attack Console
Configure your target (URL, headers, body template), import directly from cURL commands, and fire pattern-based attacks with **live streaming results**. Watch in real-time as each payload hits and see exactly how your AI responds.

![Attack Console](docs/images/attack-console.png)

### 📚 Education Tab
New to prompt injection? The built-in education tab covers:
- What prompt injection is and why it matters
- How to find testable AI endpoints
- How to interpret scan results
- Common vulnerability categories explained

**No prior security experience needed.** The onboarding walkthrough guides you from zero to your first scan.

![Education Tab](docs/images/education-tab.png)

### 🔍 Pattern Browser
Browse, search, and explore attack patterns organized by category. Each pattern includes:
- The attack payload
- What it does and why it works
- Difficulty level (beginner → advanced)
- Category (jailbreak, data extraction, instruction override, etc.)

![Pattern Browser](docs/images/pattern-browser.png)

### 🤖 LLM Verdict (Optional)
Connect a local [Ollama](https://ollama.ai) instance to get AI-powered classification of responses. Judgement will analyze whether the target was successfully exploited, partially resistant, or fully defended.

### 💾 Session History
All scan sessions and results are stored locally in SQLite. Review past scans, compare results across targets, and track your testing progress.

### 🔒 Built-in Safety
- **SSRF Protection** — Target URL validation prevents scanning internal/private networks
- **Local-only by default** — Binds to localhost, no accidental exposure
- **Zero telemetry** — Nothing phones home, ever
- **Responsible use disclaimer** — Prominent warnings on every page

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   You pick   │────▶│  Judgement    │────▶│  Your AI    │
│   patterns   │     │  fires them  │     │  endpoint   │
└─────────────┘     └──────┬───────┘     └──────┬──────┘
                           │                     │
                    ┌──────▼───────┐     ┌──────▼──────┐
                    │  Results     │◀────│  Response   │
                    │  + Verdict   │     │  captured   │
                    └──────────────┘     └─────────────┘
```

1. **Configure** — Point Judgement at your AI endpoint (URL + headers + body template)
2. **Select** — Choose attack patterns by category or difficulty
3. **Fire** — Watch results stream in real-time
4. **Analyze** — Review responses, optional LLM verdict classifies each result
5. **Fix** — Use the findings to harden your AI's defenses

## Custom Patterns

Place your patterns in `patterns.json` in the project root:

```json
{
  "id": "custom-001",
  "category": "jailbreak",
  "text": "The attack payload text...",
  "explanation": "What this pattern attempts to do",
  "why_it_works": "Why this technique is effective against LLMs",
  "difficulty": "intermediate"
}
```

**Categories:** `jailbreak`, `data_extraction`, `instruction_override`, `encoding`, `multi_turn`, `social_engineering`, `system_prompt_leak`

**Difficulty levels:** `beginner`, `intermediate`, `advanced`, `expert`

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `--port` | `8668` | Server port |
| `--host` | `127.0.0.1` | Bind address |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:14b` | Model for LLM verdict |

## OSS vs Pro

| Feature | OSS (Free) | Pro (Hosted) |
|---------|:----------:|:------------:|
| Attack console | ✅ | ✅ |
| Education tab | ✅ | ✅ |
| Pattern browser | ✅ | ✅ |
| LLM verdict | ✅ (bring your own Ollama) | ✅ (built-in) |
| Starter patterns | ✅ | ✅ |
| 240K+ curated patterns | ❌ | ✅ |
| Weekly pattern updates | ❌ | ✅ |
| Campaigns & leaderboard | ❌ | ✅ |
| MCP server integration | ✅ | ✅ |
| Multi-turn attack chains | ❌ | ✅ |
| Priority support | ❌ | ✅ |

**[Try Judgement Pro →](https://judgement.fallenangelsystems.com)**

## Contributing

Contributions are welcome! Here's how to help:

- 🐛 **Bug reports** — [Open an issue](https://github.com/fallen-angel-systems/fas-judgement-oss/issues)
- 💡 **Feature requests** — [Open an issue](https://github.com/fallen-angel-systems/fas-judgement-oss/issues) with the `enhancement` label
- 🔧 **Pull requests** — Fork, branch, PR. Keep changes focused and include a description.
- 📝 **Pattern contributions** — Submit new attack patterns via PR to `patterns.json`

## Related Projects

- **[Guardian](https://fallenangelsystems.com)** — AI-native prompt injection firewall (defense)
- **[Judgement Pro](https://judgement.fallenangelsystems.com)** — Full-featured hosted version with 240K+ patterns

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with 🔥 by [Fallen Angel Systems](https://fallenangelsystems.com)

*If Judgement found a vulnerability in your AI, imagine what an attacker would find.*

</div>

> **DISCLAIMER:** This tool is intended for authorized security testing and educational purposes only. Only test systems you own or have explicit written permission to test. Unauthorized access to computer systems is illegal under the Computer Fraud and Abuse Act (CFAA) and equivalent laws worldwide. The authors assume no liability for misuse of this tool.

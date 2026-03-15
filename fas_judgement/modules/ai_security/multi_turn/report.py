"""
Multi-Turn Session Report Generator
--------------------------------------
WHY: Separate from core/report.py — this generates session-specific HTML/PDF
     reports from multi-turn attack sessions, styled for printing/sharing.

LAYER: Module/Report — stdlib only (json, datetime).
SOURCE: Direct port of multi-turn-engine/report.py.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .service import Session, PhaseResult, PHASE_DESCRIPTIONS, CATEGORY_GOALS


def _severity_color(severity: str) -> str:
    return {
        "critical": "#ef4444",
        "high": "#f59e0b",
        "medium": "#3b82f6",
        "low": "#22c55e",
    }.get(severity.lower(), "#94a3b8")


def _score_color(score: str) -> str:
    return {
        "comply": "#ef4444",
        "partial": "#f59e0b",
        "deflect": "#94a3b8",
        "refuse": "#22c55e",
        "detect": "#06b6d4",
    }.get(score.lower() if score else "", "#94a3b8")


def _score_badge(score: str) -> str:
    color = _score_color(score)
    label = (score or "pending").upper()
    return f'<span style="background:{color}15;color:{color};border:1px solid {color}40;padding:2px 8px;border-radius:4px;font-size:9pt;font-weight:600;">{label}</span>'


def generate_html_report(session: Session) -> str:
    """Generate a styled HTML report from a session."""

    # Calculate stats
    total_turns = len(session.turns)
    scores = [t.score.value for t in session.turns if t.score]
    comply_count = scores.count("comply")
    partial_count = scores.count("partial")
    deflect_count = scores.count("deflect")
    refuse_count = scores.count("refuse")
    detect_count = scores.count("detect")
    findings_count = len(session.findings)

    # Determine overall result
    if comply_count > 0:
        overall = "VULNERABLE"
        overall_color = "#ef4444"
    elif partial_count > 0:
        overall = "PARTIALLY VULNERABLE"
        overall_color = "#f59e0b"
    elif detect_count > 0:
        overall = "DETECTED"
        overall_color = "#06b6d4"
    elif refuse_count > 0:
        overall = "RESISTANT"
        overall_color = "#22c55e"
    else:
        overall = "INCONCLUSIVE"
        overall_color = "#94a3b8"

    # Build turn rows
    turn_rows = ""
    for t in session.turns:
        score_val = t.score.value if t.score else "pending"
        turn_rows += f"""
        <div class="turn-card">
            <div class="turn-header">
                <span class="turn-num">Turn {t.turn_number}</span>
                <span class="turn-phase">Phase {t.phase}</span>
                {_score_badge(score_val)}
            </div>
            <div class="turn-attack">
                <div class="turn-label">ATTACK</div>
                <p>{_escape(t.attack_message)}</p>
            </div>
            <div class="turn-response">
                <div class="turn-label">RESPONSE</div>
                <p>{_escape(t.target_response or 'No response recorded')}</p>
            </div>
            <div class="turn-reason">
                <div class="turn-label">ANALYSIS</div>
                <p>{_escape(t.score_reason or 'Not scored')}</p>
            </div>
        </div>"""

    # Build findings rows
    findings_rows = ""
    if session.findings:
        for f in session.findings:
            sev = f.get("severity", "medium")
            sev_color = _severity_color(sev)
            findings_rows += f"""
            <div class="finding-card" style="border-left:4px solid {sev_color};">
                <div class="finding-header">
                    <span style="color:{sev_color};font-weight:700;text-transform:uppercase;font-size:9pt;">{sev}</span>
                    <span class="turn-phase">Turn {f['turn']} / Phase {f['phase']}</span>
                </div>
                <div class="turn-attack">
                    <div class="turn-label">ATTACK</div>
                    <p>{_escape(f['attack'])}</p>
                </div>
                <div class="turn-response">
                    <div class="turn-label">TARGET DISCLOSED</div>
                    <p>{_escape(f.get('response', ''))}</p>
                </div>
            </div>"""
    else:
        findings_rows = '<p style="color:#22c55e;font-weight:600;">No sensitive information was disclosed during this test.</p>'

    category_goal = CATEGORY_GOALS.get(session.category, "Extract sensitive information")
    created = session.created_at[:19].replace("T", " ") if session.created_at else "Unknown"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Judgement Elite - Attack Report</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @page {{
        size: letter;
        margin: 0.7in 0.8in;
        @bottom-center {{
            content: "FAS Judgement Elite - Confidential";
            font-family: 'Inter', sans-serif; font-size: 7pt; color: #94a3b8;
        }}
        @bottom-right {{
            content: counter(page);
            font-family: 'Inter', sans-serif; font-size: 7pt; color: #94a3b8;
        }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 10pt; line-height: 1.5; color: #1a1a2e; margin: 0; padding: 0;
    }}
    .header {{
        background: linear-gradient(135deg, #0a0a1a, #1a1a2e);
        color: white; padding: 0.5in 0.8in; margin: -0.7in -0.8in 0.4in -0.8in;
        position: relative;
    }}
    .header::after {{
        content: ''; position: absolute; bottom: 0; left: 0; right: 0;
        height: 4px; background: linear-gradient(90deg, #ec4899, #06b6d4);
    }}
    .header h1 {{
        font-size: 20pt; font-weight: 800; margin: 0 0 0.05in 0; color: #fff;
    }}
    .header .subtitle {{
        font-size: 10pt; color: #94a3b8; margin: 0;
    }}
    .header .brand {{
        font-size: 8pt; color: #64748b; margin-top: 0.15in;
        text-transform: uppercase; letter-spacing: 2px;
    }}
    .overview {{
        display: flex; gap: 0.15in; margin-bottom: 0.3in; flex-wrap: wrap;
    }}
    .stat-box {{
        flex: 1; min-width: 1.3in; background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 0.12in 0.15in; text-align: center;
    }}
    .stat-box .num {{
        font-size: 18pt; font-weight: 800; display: block;
    }}
    .stat-box .label {{
        font-size: 8pt; color: #64748b; text-transform: uppercase;
        letter-spacing: 1px; font-weight: 600;
    }}
    h2 {{
        font-size: 13pt; font-weight: 700; color: #1a1a2e;
        border-bottom: 2px solid #ec4899; padding-bottom: 0.05in;
        margin: 0.3in 0 0.15in 0;
    }}
    .meta-grid {{
        display: grid; grid-template-columns: 1fr 1fr; gap: 0.08in 0.3in;
        margin-bottom: 0.2in;
    }}
    .meta-item {{ font-size: 9.5pt; }}
    .meta-item .key {{ color: #64748b; font-weight: 600; }}
    .meta-item .val {{ color: #1a1a2e; }}
    .turn-card {{
        border: 1px solid #e2e8f0; border-radius: 8px;
        margin: 0.12in 0; page-break-inside: avoid; overflow: hidden;
    }}
    .turn-header {{
        background: #f8fafc; padding: 0.08in 0.15in;
        display: flex; align-items: center; gap: 0.1in;
        border-bottom: 1px solid #e2e8f0;
    }}
    .turn-num {{
        font-weight: 700; font-size: 10pt; color: #1a1a2e;
    }}
    .turn-phase {{
        font-size: 8pt; color: #64748b; font-weight: 600;
        background: #e2e8f0; padding: 1px 6px; border-radius: 3px;
    }}
    .turn-attack, .turn-response, .turn-reason {{
        padding: 0.06in 0.15in;
    }}
    .turn-label {{
        font-size: 7pt; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1.5px; color: #94a3b8; margin-bottom: 0.02in;
    }}
    .turn-attack p, .turn-response p, .turn-reason p {{
        margin: 0; font-size: 9.5pt; color: #334155;
    }}
    .turn-attack {{ background: #fef2f2; }}
    .turn-response {{ background: #f0f9ff; }}
    .finding-card {{
        border: 1px solid #e2e8f0; border-radius: 0 8px 8px 0;
        margin: 0.1in 0; page-break-inside: avoid; overflow: hidden;
    }}
    .finding-header {{
        padding: 0.06in 0.15in; display: flex; align-items: center;
        gap: 0.1in; background: #f8fafc; border-bottom: 1px solid #e2e8f0;
    }}
    .overall-badge {{
        display: inline-block; padding: 0.06in 0.2in; border-radius: 6px;
        font-size: 14pt; font-weight: 800; color: white;
    }}
    .footer {{
        text-align: center; color: #94a3b8; font-size: 8pt;
        margin-top: 0.4in; padding-top: 0.15in;
        border-top: 1px solid #e2e8f0;
    }}
</style>
</head>
<body>

<div class="header">
    <h1>Multi-Turn Attack Report</h1>
    <p class="subtitle">{_escape(session.category_name)} — {_escape(session.target_name or 'Target')}</p>
    <p class="brand">Fallen Angel Systems — Judgement Elite</p>
</div>

<div class="overview">
    <div class="stat-box">
        <span class="num" style="color:{overall_color};">{overall}</span>
        <span class="label">Result</span>
    </div>
    <div class="stat-box">
        <span class="num">{total_turns}</span>
        <span class="label">Turns</span>
    </div>
    <div class="stat-box">
        <span class="num" style="color:#ef4444;">{findings_count}</span>
        <span class="label">Findings</span>
    </div>
    <div class="stat-box">
        <span class="num">{session.current_phase}/{session.max_phases}</span>
        <span class="label">Phases</span>
    </div>
</div>

<h2>Test Details</h2>
<div class="meta-grid">
    <div class="meta-item"><span class="key">Session ID:</span> <span class="val">{session.id[:12]}...</span></div>
    <div class="meta-item"><span class="key">Category:</span> <span class="val">{_escape(session.category_name)}</span></div>
    <div class="meta-item"><span class="key">Mode:</span> <span class="val">{session.mode.value.upper()}</span></div>
    <div class="meta-item"><span class="key">Target:</span> <span class="val">{_escape(session.target_name or session.target_url or 'Local')}</span></div>
    <div class="meta-item"><span class="key">Started:</span> <span class="val">{created}</span></div>
    <div class="meta-item"><span class="key">Status:</span> <span class="val">{session.status.upper()}</span></div>
</div>

<p style="font-size:9.5pt;color:#475569;"><strong>Attack Goal:</strong> {_escape(category_goal)}</p>

<h2>Findings</h2>
{findings_rows}

<h2>Score Summary</h2>
<div class="overview">
    <div class="stat-box"><span class="num" style="color:#ef4444;">{comply_count}</span><span class="label">Comply</span></div>
    <div class="stat-box"><span class="num" style="color:#f59e0b;">{partial_count}</span><span class="label">Partial</span></div>
    <div class="stat-box"><span class="num" style="color:#94a3b8;">{deflect_count}</span><span class="label">Deflect</span></div>
    <div class="stat-box"><span class="num" style="color:#22c55e;">{refuse_count}</span><span class="label">Refuse</span></div>
    <div class="stat-box"><span class="num" style="color:#06b6d4;">{detect_count}</span><span class="label">Detect</span></div>
</div>

<h2>Turn-by-Turn Log</h2>
{turn_rows}

<div class="footer">
    Generated by FAS Judgement Elite — fallenangelsystems.com<br>
    © 2026 Fallen Angel Systems LLC. All rights reserved.
</div>

</body>
</html>"""

    return html


def save_html_report(session: Session, output_path: Path) -> Path:
    """Generate and save HTML report."""
    html = generate_html_report(session)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def save_pdf_report(session: Session, output_path: Path) -> Path:
    """Generate and save PDF report (requires weasyprint)."""
    html = generate_html_report(session)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(output_path))
    except ImportError:
        # Fallback: save as HTML with .pdf note
        fallback = output_path.with_suffix(".html")
        fallback.write_text(html, encoding="utf-8")
        raise ImportError(
            f"weasyprint not installed. HTML saved to {fallback}. "
            "Install with: pip install weasyprint"
        )

    return output_path


def _escape(text: str) -> str:
    """HTML-escape text."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

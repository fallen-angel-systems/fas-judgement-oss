"""
Multi-Turn API Routes
----------------------
WHY: Routes live inside the module rather than in http/ because they're
     tightly coupled to the multi-turn domain. The module's get_router()
     method mounts these at /api/multi-turn.

LAYER: Module/HTTP — imports FastAPI. The only place FastAPI appears in
       this module.
SOURCE: Direct port of multi-turn-engine/routes.py.
"""

import json
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .service import Orchestrator, Mode, PhaseResult
from .scorer import check_ollama, llm_score, keyword_score
from ....transport import HTTPTransport, OllamaTransport, create_transport
from .report import generate_html_report, save_pdf_report

router = APIRouter(tags=["multi-turn"])

# Single orchestrator instance (persists across requests)
orch = Orchestrator()

# Transport instances per session (kept alive for conversation context)
session_transports: dict = {}  # session_id -> Transport instance


# --- Connection Gate ---

@router.post("/check-connection")
async def check_connection(request: Request):
    """Check Ollama connection and return available models.
    This is the gate - multi-turn tab won't unlock without a valid connection.
    """
    body = await request.json()
    ollama_url = body.get("ollama_url", "http://localhost:11434")
    result = await check_ollama(ollama_url)
    return JSONResponse(content=result)


# --- Transport Testing ---

@router.post("/test-transport")
async def test_transport(request: Request):
    """Test a transport connection before starting a session.
    Accepts the same transport config object the UI sends on session create.
    """
    body = await request.json()
    transport_config = body.get("transport") or body
    try:
        transport = create_transport(transport_config)
        result = await transport.check_connection()
        return JSONResponse(content=result)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"connected": False, "error": str(e)})
    except Exception as e:
        return JSONResponse(content={"connected": False, "error": str(e)})


# --- Categories ---

@router.get("/categories")
async def list_categories():
    """List all available attack categories."""
    return JSONResponse(content={"categories": orch.list_categories()})


# --- Session Management ---

@router.get("/sessions")
async def list_sessions(status: str = None, limit: int = 50):
    """List all sessions (optionally filter by status)."""
    sessions = orch.list_sessions(status=status, limit=limit)
    return JSONResponse(content={"sessions": sessions})


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its data."""
    if orch.delete_session(session_id):
        return JSONResponse(content={"deleted": True})
    return JSONResponse(status_code=404, content={"error": "Session not found"})


@router.post("/sessions")
async def create_session(request: Request):
    """Start a new multi-turn attack session."""
    body = await request.json()

    category = body.get("category")
    if not category:
        return JSONResponse(
            status_code=400,
            content={"error": "category is required"},
        )

    mode_str = body.get("mode", "auto")
    try:
        mode = Mode(mode_str)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid mode: {mode_str}. Use: auto, coop, manual"},
        )

    session = orch.start_session(
        category=category,
        mode=mode,
        target_url=body.get("target_url"),
        target_name=body.get("target_name"),
        ollama_url=body.get("ollama_url", "http://localhost:11434"),
        ollama_model=body.get("ollama_model", ""),
    )

    # Create and store transport if config provided
    transport_config = body.get("transport")
    if transport_config:
        try:
            transport = create_transport(transport_config)
            session_transports[session.id] = transport
        except Exception as e:
            print(f"[multi-turn] Transport create error: {e}")

    return JSONResponse(content={
        "session_id": session.id,
        "category": session.category_name,
        "mode": session.mode.value,
        "phase": session.current_phase,
        "status": session.status,
        "has_transport": session.id in session_transports,
    })


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get current session state."""
    session = orch.get_session(session_id)
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    return JSONResponse(content=session.to_dict())


@router.get("/sessions/{session_id}/export")
async def export_session(session_id: str):
    """Export full session data as JSON."""
    try:
        data = orch.export_session(session_id)
        return JSONResponse(content=data)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})


@router.get("/sessions/{session_id}/report")
async def get_report(session_id: str, format: str = "html"):
    """Generate a styled HTML or PDF report for a session.

    Query params:
        format: "html" (default) or "pdf"
    """
    from fastapi.responses import HTMLResponse, FileResponse
    import tempfile
    from pathlib import Path

    session = orch.get_session(session_id)
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    if format == "pdf":
        try:
            tmp = Path(tempfile.mktemp(suffix=".pdf"))
            save_pdf_report(session, tmp)
            return FileResponse(
                str(tmp),
                media_type="application/pdf",
                filename=f"judgement-report-{session_id[:8]}.pdf",
            )
        except ImportError as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    else:
        html = generate_html_report(session)
        return HTMLResponse(content=html)


# --- Attack Flow ---

@router.post("/sessions/{session_id}/next")
async def next_turn(session_id: str, request: Request):
    """Pick the next attack message and optionally send it to the target.

    Body:
        custom_message (optional): Override with a manual message (coop/manual mode)
        auto_send (optional): If true, send the message to the target and score the response
        target_url (optional): Override target URL for this turn
    """
    session = orch.get_session(session_id)
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    body = await request.json() if await request.body() else {}
    custom_message = body.get("custom_message")
    auto_send = body.get("auto_send", False)

    try:
        turn = orch.pick_message(session_id, custom_message=custom_message)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    result = {
        "turn_number": turn.turn_number,
        "phase": turn.phase,
        "attack_message": turn.attack_message,
        "session_status": session.status,
    }

    # Auto-send through transport if one is configured for this session
    transport = session_transports.get(session_id)
    if transport or auto_send:
        try:
            # Use stored transport, or fall back to creating one from body/session
            if not transport and session.target_url:
                transport = HTTPTransport(
                    target_url=body.get("target_url", session.target_url),
                    body_field=body.get("body_field", "message"),
                    response_field=body.get("response_field", "response"),
                )

            if transport:
                target_response = await transport.send(turn.attack_message)

                # Build history summary for phase-aware scoring
                history_lines = []
                for t in session.turns[:-1]:  # all turns except current
                    if t.score:
                        history_lines.append(f"Turn {t.turn_number} (phase {t.phase}): {t.score.value}")
                history_summary = "; ".join(history_lines) if history_lines else "First turn."

                # Score with LLM first, keyword fallback
                if session.ollama_model:
                    score, reason = await llm_score(
                        turn.attack_message,
                        target_response,
                        ollama_url=session.ollama_url,
                        model=session.ollama_model,
                        category=session.category,
                        phase=session.current_phase,
                        max_phases=session.max_phases,
                        turn_number=turn.turn_number,
                        history_summary=history_summary,
                    )
                else:
                    score, reason = keyword_score(target_response)

                action = orch.record_response(
                    session_id,
                    response=target_response,
                    score=score,
                    score_reason=reason,
                )

                result["target_response"] = target_response
                result["score"] = score.value
                result["score_reason"] = reason
                result["action"] = action
                result["auto_sent"] = True

        except Exception as e:
            result["send_error"] = str(e)
            result["action"] = {"action": "error", "message": f"Transport error: {e}"}

    return JSONResponse(content=result)


@router.post("/sessions/{session_id}/score")
async def score_response(session_id: str, request: Request):
    """Score a target response (when not using auto_send).

    Body:
        response: The target's response text
        manual_score (optional): Override with manual score (1-5 or name)
        manual_reason (optional): Reason for manual score
    """
    session = orch.get_session(session_id)
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    body = await request.json()
    response = body.get("response", "")

    if not response:
        return JSONResponse(status_code=400, content={"error": "response is required"})

    # Manual score override
    manual_score = body.get("manual_score")
    if manual_score:
        score_map = {
            "1": PhaseResult.COMPLY, "comply": PhaseResult.COMPLY,
            "2": PhaseResult.PARTIAL, "partial": PhaseResult.PARTIAL,
            "3": PhaseResult.DEFLECT, "deflect": PhaseResult.DEFLECT,
            "4": PhaseResult.REFUSE, "refuse": PhaseResult.REFUSE,
            "5": PhaseResult.DETECT, "detect": PhaseResult.DETECT,
        }
        score = score_map.get(str(manual_score).lower(), PhaseResult.DEFLECT)
        reason = body.get("manual_reason", f"Manual score: {manual_score}")
    elif session.ollama_model:
        # LLM scoring (phase-aware)
        last_turn = session.turns[-1] if session.turns else None
        attack_msg = last_turn.attack_message if last_turn else ""

        history_lines = []
        for t in session.turns[:-1]:
            if t.score:
                history_lines.append(f"Turn {t.turn_number} (phase {t.phase}): {t.score.value}")
        history_summary = "; ".join(history_lines) if history_lines else "First turn."

        score, reason = await llm_score(
            attack_msg,
            response,
            ollama_url=session.ollama_url,
            model=session.ollama_model,
            category=session.category,
            phase=session.current_phase,
            max_phases=session.max_phases,
            turn_number=last_turn.turn_number if last_turn else 1,
            history_summary=history_summary,
        )
    else:
        # Keyword fallback
        score, reason = keyword_score(response)

    action = orch.record_response(
        session_id,
        response=response,
        score=score,
        score_reason=reason,
    )

    return JSONResponse(content={
        "score": score.value,
        "reason": reason,
        "action": action,
        "session_status": session.status,
        "phase": session.current_phase,
    })


# --- Co-op Controls ---

@router.post("/sessions/{session_id}/pause")
async def pause_session(session_id: str):
    """Pause a session (Hold in Hold & Inject)."""
    if orch.pause_session(session_id):
        return JSONResponse(content={"status": "paused"})
    return JSONResponse(status_code=400, content={"error": "Cannot pause - session not active"})


@router.post("/sessions/{session_id}/resume")
async def resume_session(session_id: str):
    """Resume a paused session."""
    if orch.resume_session(session_id):
        return JSONResponse(content={"status": "active"})
    return JSONResponse(status_code=400, content={"error": "Cannot resume - session not paused"})


@router.get("/sessions/{session_id}/suggest")
async def get_suggestions(session_id: str):
    """Get orchestrator suggestions for next move (coop/manual mode)."""
    try:
        suggestions = orch.get_suggestion(session_id)
        return JSONResponse(content=suggestions)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})


# --- Ollama Target Testing ---

@router.post("/sessions/{session_id}/test-local")
async def test_against_local(session_id: str, request: Request):
    """Run attack against a local Ollama model (for testing/demos).

    Body:
        model: Ollama model name to attack
        system_prompt: System prompt for the target model
        ollama_url (optional): Ollama URL (defaults to session's)
    """
    session = orch.get_session(session_id)
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    body = await request.json()
    target_model = body.get("model", "qwen2.5:14b")
    system_prompt = body.get("system_prompt", "You are a helpful assistant.")
    ollama_url = body.get("ollama_url", session.ollama_url)

    # Pick next message
    try:
        turn = orch.pick_message(session_id)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    # Send to local model
    transport = OllamaTransport(
        ollama_url=ollama_url,
        model=target_model,
        system_prompt=system_prompt,
    )

    try:
        target_response = await transport.send(turn.attack_message)
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": f"Failed to reach Ollama: {e}",
            "attack_message": turn.attack_message,
        })

    # Score the response
    scoring_model = session.ollama_model or "qwen2.5:14b"
    # Use a different model for scoring if same as target
    if scoring_model == target_model:
        score, reason = keyword_score(target_response)
        reason = f"Keyword scoring (scorer model same as target): {reason}"
    else:
        score, reason = await llm_score(
            turn.attack_message,
            target_response,
            ollama_url=ollama_url,
            model=scoring_model,
        )

    action = orch.record_response(
        session_id,
        response=target_response,
        score=score,
        score_reason=reason,
    )

    return JSONResponse(content={
        "turn_number": turn.turn_number,
        "phase": turn.phase,
        "attack_message": turn.attack_message,
        "target_response": target_response,
        "score": score.value,
        "score_reason": reason,
        "action": action,
        "target_model": target_model,
    })

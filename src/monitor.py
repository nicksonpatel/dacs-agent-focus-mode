"""Terminal monitor for DACS experiments.

Hooks into Logger as a sink and prints a rich, colour-coded live feed of every
event to the console as it happens.

Output format (one or more lines per event):

  12:00:01 │ ▶ RUN START    s1_n3 · DACS · 3 agents · MiniMax-M2.7
  12:00:01 │ ⇢ REQUEST      [a3] urgency=HIGH  "Description column has mixed UTF-8/Latin-1…"
  12:00:01 │ ◈ CONTEXT      FOCUS  a3  →  281 tokens
  12:00:02 │ ◉ LLM CALL     FOCUS  a3  │  in=281  out=454  │  10 268 ms
  12:00:02 │ ✔ STEERING     [a3]  ctx=281  │  Q: Description column has mixed…
           │                              │  A: Coerce all rows to UTF-8. Use…
  12:00:02 │ → TRANSITION   FOCUS(a3) → REGISTRY   [SteeringComplete]
  12:00:02 │ · REGISTRY     a3  RUNNING  LOW  (87 tok)
  ...
  12:05:01 │ ■ RUN END      9 steering  ·  9 LLM calls  ·  47.2 s
"""
from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console
from rich.text import Text

_console = Console(highlight=False)

# ANSI-safe colour palette (rich style strings)
_STYLE = {
    "RUN_START":        "bold green",
    "RUN_END":          "bold green",
    "TRANSITION":       "bold cyan",
    "STEERING_REQUEST": "bold yellow",
    "CONTEXT_BUILT":    "dim white",
    "LLM_CALL":        "bold blue",
    "STEERING_RESPONSE":"bold magenta",
    "REGISTRY_UPDATE":  "dim",
    "REGISTRY_TRUNCATION": "bold red",
    "INTERRUPT":        "bold red",
    "FOCUS_TIMEOUT":    "bold red",
}

_BADGE = {
    "RUN_START":         "▶  RUN START   ",
    "RUN_END":           "■  RUN END     ",
    "TRANSITION":        "→  TRANSITION  ",
    "STEERING_REQUEST":  "⇢  REQUEST     ",
    "CONTEXT_BUILT":     "◈  CONTEXT     ",
    "LLM_CALL":          "◉  LLM CALL    ",
    "STEERING_RESPONSE": "✔  STEERING    ",
    "REGISTRY_UPDATE":   "·  REGISTRY    ",
    "REGISTRY_TRUNCATION":"⚠  TRUNCATION  ",
    "INTERRUPT":         "⚡ INTERRUPT   ",
    "FOCUS_TIMEOUT":     "⏱  TIMEOUT     ",
}

_CONTINUATION = "           │               "


def _ts(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%H:%M:%S")
    except Exception:
        return "??:??:??"


def _trunc(s: str, n: int = 80) -> str:
    s = s.replace("\n", " ").strip()
    return s[:n] + "…" if len(s) > n else s


def _token_bar(n: int, max_n: int = 204800, width: int = 20) -> str:
    filled = max(1, round(n / max_n * width))
    return "█" * filled + "░" * (width - filled)


class TerminalMonitor:
    """Register with Logger.add_sink(monitor.handle) to activate live output."""

    def __init__(self, token_budget: int = 204800) -> None:
        self._budget  = token_budget
        self._llm_count = 0
        self._steering_count = 0

    def handle(self, event: dict) -> None:
        etype  = event.get("event", "UNKNOWN")
        ts     = _ts(event.get("ts", ""))
        badge  = _BADGE.get(etype, f"?  {etype:<13}")
        style  = _STYLE.get(etype, "white")
        prefix = f"[dim]{ts}[/dim] [dim]│[/dim] "

        if etype == "RUN_START":
            cond  = "DACS" if event.get("focus_mode") else "BASELINE"
            line  = (
                f"{prefix}[{style}]{badge}[/{style}]"
                f"[bold]{event.get('scenario')}[/bold]  ·  "
                f"[bold]{cond}[/bold]  ·  "
                f"{event.get('n_agents')} agents  ·  "
                f"[dim]{event.get('model')}[/dim]  ·  "
                f"budget={event.get('token_budget'):,}"
            )
            _console.rule(f"[green]Run {event.get('run_id')}[/green]")
            _console.print(line)

        elif etype == "RUN_END":
            line = (
                f"{prefix}[{style}]{badge}[/{style}]"
                f"[bold]{self._steering_count}[/bold] steering  ·  "
                f"[bold]{self._llm_count}[/bold] LLM calls"
            )
            _console.print(line)
            _console.rule()

        elif etype == "TRANSITION":
            frm  = event.get("from", "?")
            to   = event.get("to", "?")
            aid  = event.get("agent_id")
            trig = event.get("trigger", "")
            to_label = f"FOCUS({aid})" if to == "FOCUS" and aid else to
            frm_label = f"FOCUS({self._focus_agent})" if frm == "FOCUS" and hasattr(self, "_focus_agent") and self._focus_agent else frm
            if to == "FOCUS":
                self._focus_agent = aid
            elif frm == "FOCUS":
                self._focus_agent = None
            line = (
                f"{prefix}[{style}]{badge}[/{style}]"
                f"[cyan]{frm_label}[/cyan] → [cyan]{to_label}[/cyan]"
                f"   [dim]{trig}[/dim]"
            )
            _console.print(line)

        elif etype == "STEERING_REQUEST":
            aid     = event.get("agent_id", "?")
            urgency = event.get("urgency", "?")
            urg_col = {"HIGH": "bold red", "MEDIUM": "yellow", "LOW": "dim"}.get(urgency, "white")
            line = (
                f"{prefix}[{style}]{badge}[/{style}]"
                f"[bold]\\[{aid}][/bold]  urgency=[{urg_col}]{urgency}[/{urg_col}]  "
                f"blocking={event.get('blocking')}"
            )
            _console.print(line)

        elif etype == "CONTEXT_BUILT":
            mode    = event.get("mode", "?")
            aid     = event.get("agent_id") or "—"
            tokens  = event.get("token_count", 0)
            bar     = _token_bar(tokens, self._budget, 16)
            mode_col = {"FOCUS": "magenta", "REGISTRY": "green", "FLAT": "red"}.get(mode, "white")
            line = (
                f"{prefix}[{style}]{badge}[/{style}]"
                f"[{mode_col}]{mode:<8}[/{mode_col}]  "
                f"agent=[bold]{aid}[/bold]  "
                f"[cyan]{tokens:>6,}[/cyan] tokens  [dim]{bar}[/dim]"
            )
            _console.print(line)

        elif etype == "LLM_CALL":
            self._llm_count += 1
            state   = event.get("state", "?")
            aid     = event.get("agent_id") or "—"
            in_tok  = event.get("context_tokens", 0)
            out_tok = event.get("response_tokens", 0)
            ms      = event.get("latency_ms", 0)
            ms_str  = f"{ms:,} ms" if ms < 10000 else f"{ms/1000:.1f} s"
            line = (
                f"{prefix}[{style}]{badge}[/{style}]"
                f"[cyan]{state:<8}[/cyan]  agent=[bold]{aid}[/bold]  "
                f"│  in=[cyan]{in_tok:,}[/cyan]  out=[cyan]{out_tok:,}[/cyan]  "
                f"│  [dim]{ms_str}[/dim]"
            )
            _console.print(line)

        elif etype == "STEERING_RESPONSE":
            self._steering_count += 1
            aid     = event.get("agent_id", "?")
            ctx_tok = event.get("context_size_at_time", 0)
            mode    = event.get("orchestrator_state", "?")
            resp    = _trunc(event.get("response_text", ""), 100)
            line = (
                f"{prefix}[{style}]{badge}[/{style}]"
                f"[bold]\\[{aid}][/bold]  ctx={ctx_tok:,} tok  mode={mode}"
            )
            _console.print(line)
            _console.print(
                f"{_CONTINUATION}  [magenta]↳[/magenta] [italic]{resp}[/italic]"
            )

        elif etype == "REGISTRY_UPDATE":
            aid     = event.get("agent_id", "?")
            status  = event.get("status", "?")
            urgency = event.get("urgency", "?")
            stok    = event.get("summary_tokens", 0)
            urg_col = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "dim"}.get(urgency, "white")
            line = (
                f"{prefix}[{style}]{badge}[/{style}]"
                f"[dim][bold]{aid}[/bold]  {status:<18}  "
                f"[{urg_col}]{urgency}[/{urg_col}]  ({stok} tok)[/dim]"
            )
            _console.print(line)

        elif etype == "INTERRUPT":
            line = (
                f"{prefix}[{style}]{badge}[/{style}]"
                f"[red]pausing [bold]{event.get('interrupted_agent')}[/bold]  →  "
                f"switching to [bold]{event.get('interrupting_agent')}[/bold][/red]  "
                f"[dim](HIGH urgency)[/dim]"
            )
            _console.print(line)

        elif etype == "FOCUS_TIMEOUT":
            line = (
                f"{prefix}[{style}]{badge}[/{style}]"
                f"[red][bold]{event.get('agent_id')}[/bold]  "
                f"elapsed={event.get('elapsed_s')}s  turns={event.get('turns')}[/red]"
            )
            _console.print(line)

        elif etype == "REGISTRY_TRUNCATION":
            line = (
                f"{prefix}[{style}]{badge}[/{style}]"
                f"[red]{event.get('agent_id')}  "
                f"summary truncated: {event.get('original_tokens')} → 100 tokens[/red]"
            )
            _console.print(line)

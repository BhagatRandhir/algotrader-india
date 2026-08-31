"""utils/dashboard.py – Live Rich terminal dashboard."""
from __future__ import annotations
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def render_dashboard(
    cash: float,
    daily_pnl: float,
    positions: dict,
    signal_log: list[dict],
    halted: bool,
    halt_reason: str = "",
    loop_count: int = 0,
    market_open: bool = True,
):
    console.clear()

    # Header
    status = "🛑 HALTED" if halted else ("🟢 LIVE" if market_open else "🌙 MARKET CLOSED")
    pnl_color = "green" if daily_pnl >= 0 else "red"
    pnl_sign  = "+" if daily_pnl >= 0 else ""

    console.print(Panel(
        f"  AlgoTrader India  |  {status}  |  "
        f"Cash: [cyan]₹{cash:,.0f}[/]  |  "
        f"Day P&L: [{pnl_color}]{pnl_sign}₹{daily_pnl:,.0f}[/]  |  "
        f"Loop #{loop_count}  |  {datetime.now().strftime('%d %b %Y  %H:%M:%S IST')}",
        style="bold white on black",
        box=box.DOUBLE_EDGE,
    ))

    if halted:
        console.print(f"\n  [red bold]Halt reason: {halt_reason}[/]\n")

    # Positions
    pos_tbl = Table(title="Open Positions", box=box.SIMPLE_HEAVY,
                    title_style="bold cyan", style="cyan")
    pos_tbl.add_column("Symbol", style="bold white")
    pos_tbl.add_column("Qty", justify="right")
    pos_tbl.add_column("Avg Price", justify="right")
    pos_tbl.add_column("P&L", justify="right")
    pos_tbl.add_column("Product")

    if positions:
        for sym, p in positions.items():
            pnl = p.get("pnl", 0)
            pnl_str = f"[{'green' if pnl >= 0 else 'red'}]₹{pnl:+,.0f}[/]"
            pos_tbl.add_row(
                sym, str(p["qty"]),
                f"₹{p['average_price']:.2f}",
                pnl_str, p.get("product", ""),
            )
    else:
        pos_tbl.add_row("[dim]No open positions[/]", "", "", "", "")

    console.print(pos_tbl)

    # Signal log
    sig_tbl = Table(title="Signal Log (last 10)", box=box.SIMPLE_HEAVY,
                    title_style="bold yellow", style="yellow")
    sig_tbl.add_column("Time", style="dim", width=10)
    sig_tbl.add_column("Symbol", style="bold white", width=12)
    sig_tbl.add_column("Signal", width=6)
    sig_tbl.add_column("Score", justify="right", width=6)
    sig_tbl.add_column("Votes", width=40)

    colors = {"BUY": "green", "SELL": "red", "HOLD": "dim"}
    for entry in signal_log[-10:]:
        sig = entry.get("signal", "HOLD")
        votes = "  ".join(
            f"[{colors.get(v,'white')}]{k[:3]}:{v[0]}[/]"
            for k, v in entry.get("votes", {}).items()
        )
        sig_tbl.add_row(
            entry.get("time", ""),
            entry.get("symbol", ""),
            f"[{colors.get(sig,'white')}]{sig}[/]",
            f"{entry.get('strength', 0):.2f}",
            votes,
        )

    console.print(sig_tbl)
    console.print("\n  [dim]Ctrl+C to stop  |  Logs → logs/trader.log[/]\n")


def log_trade(action: str, symbol: str, qty: int, price: float,
              sl: float, target: float, order_id: str):
    color = "green" if action == "BUY" else "red"
    console.print(
        f"\n  [{color} bold]{action}[/]  {symbol}  "
        f"qty={qty}  @₹{price:.2f}  "
        f"SL=₹{sl:.2f}  Target=₹{target:.2f}  "
        f"[dim]order={order_id}[/]\n"
    )

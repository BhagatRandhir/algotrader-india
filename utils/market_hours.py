"""
utils/market_hours.py  –  NSE/BSE market session checks (IST).

NSE/BSE trading hours:
  Pre-open   : 09:00 – 09:15 IST
  Normal     : 09:15 – 15:30 IST
  MIS sq-off : ~15:15 IST (Zerodha auto-squares MIS at 15:20)
  Closed     : weekends + NSE holidays
"""
from __future__ import annotations
from datetime import time, datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")

# NSE public holidays 2025 (add more as needed)
NSE_HOLIDAYS_2025 = {
    "2025-01-26",  # Republic Day
    "2025-03-14",  # Holi
    "2025-04-14",  # Dr. Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # Maharashtra Day
    "2025-08-15",  # Independence Day
    "2025-10-02",  # Gandhi Jayanti
    "2025-10-24",  # Dussehra
    "2025-11-05",  # Diwali Laxmi Puja
    "2025-11-14",  # Gurunanak Jayanti
    "2025-12-25",  # Christmas
}

def now_ist() -> datetime:
    return datetime.now(IST)

def is_market_open() -> bool:
    """Return True if NSE normal session is active right now."""
    now = now_ist()
    if now.weekday() >= 5:                          # Sat / Sun
        return False
    date_str = now.strftime("%Y-%m-%d")
    if date_str in NSE_HOLIDAYS_2025:
        return False
    t = now.time()
    return time(9, 15) <= t <= time(15, 30)

def is_safe_to_enter() -> bool:
    """Avoid new entries in last 15 min (MIS sq-off risk)."""
    now = now_ist()
    t = now.time()
    return is_market_open() and t <= time(15, 10)

def minutes_to_open() -> int:
    """Minutes until market opens (0 if already open)."""
    now = now_ist()
    if is_market_open():
        return 0
    open_today = now.replace(hour=9, minute=15, second=0, microsecond=0)
    delta = (open_today - now).total_seconds() / 60
    return max(int(delta), 0)

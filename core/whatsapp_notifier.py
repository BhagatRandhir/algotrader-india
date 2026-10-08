import os, urllib.parse, urllib.request
from loguru import logger

CALLMEBOT_URL = "https://api.callmebot.com/whatsapp.php"

class WhatsAppNotifier:
    def __init__(self):
        phones   = [p.strip() for p in os.getenv("WHATSAPP_PHONE","").split(",") if p.strip()]
        api_keys = [k.strip() for k in os.getenv("WHATSAPP_APIKEY","").split(",") if k.strip()]
        if len(api_keys) == 1: api_keys = api_keys * len(phones)
        self.recipients = list(zip(phones, api_keys))
        self.enabled    = bool(self.recipients)
        if self.enabled:
            logger.info(f"✅ WhatsApp enabled → {len(self.recipients)} recipient(s)")
        else:
            logger.warning("⚠️ WhatsApp not configured — add WHATSAPP_PHONE and WHATSAPP_APIKEY to .env")

    def send(self, message: str) -> bool:
        if not self.enabled: return False
        success = False
        for phone, api_key in self.recipients:
            try:
                params = urllib.parse.urlencode({"phone":phone,"text":message,"apikey":api_key})
                req = urllib.request.Request(f"{CALLMEBOT_URL}?{params}", headers={"User-Agent":"Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200 and "Message Sent" in resp.read().decode():
                        success = True
            except Exception as exc:
                logger.warning(f"WhatsApp failed: {exc}")
        return success

    def notify_buy(self, symbol, qty, entry, sl, target, strength=0.0):
        self.send(f"✅ *BUY {symbol}*\nQty: {qty}\nEntry: ₹{entry:.2f}\nSL: ₹{sl:.2f}\nTarget: ₹{target:.2f}\nStrength: {strength:.0%}")

    def notify_sell(self, symbol, qty, entry, exit_px, pnl, reason=""):
        pnl_pct = (exit_px-entry)/entry*100
        icon = "💰" if pnl>=0 else "🔴"
        self.send(f"{icon} *SELL {symbol}*\nQty: {qty}\n₹{entry:.2f} → ₹{exit_px:.2f}\nP&L: {'+'if pnl>=0 else ''}₹{pnl:.0f} ({pnl_pct:+.2f}%)\n{reason}")

    def notify_alert(self, message: str):
        self.send(f"🚨 *AlgoTrader Alert*\n{message}")

    def notify_daily_summary(self, trades, wins, losses, total_pnl, nav):
        icon = "📈" if total_pnl>=0 else "📉"
        self.send(f"{icon} *Daily Summary*\nTrades: {trades} ({wins}W/{losses}L)\nP&L: {'+'if total_pnl>=0 else ''}₹{total_pnl:.0f}\nNAV: ₹{nav:,.0f}")

_notifier = None
def get_notifier():
    global _notifier
    if _notifier is None: _notifier = WhatsAppNotifier()
    return _notifier

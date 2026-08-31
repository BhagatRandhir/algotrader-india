"""
auth/login.py  –  Generate a fresh Zerodha access token each morning.

IMPORTANT: Kite Connect access tokens expire at midnight IST.
Run this script once before the market opens (9:00–9:15 AM IST).

Usage:
    python auth/login.py

It will:
  1. Open the Kite login URL in your browser
  2. Ask you to paste the request_token from the redirect URL
  3. Exchange it for an access_token
  4. Write the token to .env automatically

⚠️  From April 2025, Zerodha requires a STATIC IP to place orders.
    Make sure your machine's IP is whitelisted in the Kite Developer Console.
"""

import os
import re
import webbrowser
from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect

ENV_FILE = os.path.join(os.path.dirname(__file__), "..", ".env")

load_dotenv(ENV_FILE)

API_KEY    = os.getenv("ZERODHA_API_KEY")
API_SECRET = os.getenv("ZERODHA_API_SECRET")

if not API_KEY or not API_SECRET:
    raise SystemExit(
        "❌  ZERODHA_API_KEY and ZERODHA_API_SECRET must be set in .env first."
    )

kite = KiteConnect(api_key=API_KEY)

login_url = kite.login_url()
print(f"\n🔗  Opening Kite login in your browser…\n    {login_url}\n")
webbrowser.open(login_url)

print(
    "After logging in, Zerodha will redirect you to your redirect URL.\n"
    "The URL will look like:\n"
    "  https://your-redirect-url/?request_token=XXXXXX&action=login&status=success\n"
)

redirect_url = input("Paste the full redirect URL here: ").strip()

# Extract request_token from URL
match = re.search(r"request_token=([^&]+)", redirect_url)
if not match:
    raise SystemExit("❌  Could not find request_token in the URL. Try again.")

request_token = match.group(1)
print(f"\n✅  request_token extracted: {request_token[:8]}…")

# Generate access token
try:
    session = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = session["access_token"]
except Exception as e:
    raise SystemExit(f"❌  Failed to generate session: {e}")

# Save to .env
set_key(ENV_FILE, "ZERODHA_ACCESS_TOKEN", access_token)
print(f"\n✅  Access token saved to .env  ({access_token[:8]}…)")
print("    You can now run:  python main.py\n")

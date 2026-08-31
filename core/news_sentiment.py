"""
core/news_sentiment.py  –  Priority 3: News Sentiment Filter

Sources (all free, no API key needed):
  - Economic Times Markets RSS
  - Moneycontrol Markets RSS
  - NSE official announcements RSS
  - Reuters India RSS
  - Google News RSS (filtered per symbol)

Pipeline:
  1. Fetch RSS headlines for each symbol + general market
  2. Score each headline with VADER (financial-tuned lexicon)
  3. Boost/penalise score for financial keywords
     (e.g. "fraud", "promoter selling" → heavy negative)
  4. Aggregate into per-symbol SentimentResult
  5. Gate:
       POSITIVE  (score > +0.15) → allow entry, full size
       NEUTRAL   (-0.15 to +0.15) → allow entry, 75% size
       NEGATIVE  (score < -0.15) → block BUY entry for the day

Result is cached per symbol for 30 minutes to avoid re-fetching.
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ── Financial keyword boosters ────────────────────────────────────
# Multipliers applied ON TOP of VADER score

# Words that VADER over-inflates in financial headlines — dampen them
DAMPEN_WORDS = {
    "decision", "policy", "meeting", "scheduled", "expected",
    "watch", "flat", "await", "ahead", "upcoming", "review",
}

NEGATIVE_BOOST = {
    "fraud": -0.6, "scam": -0.6, "cbi": -0.5, "ed raid": -0.7,
    "sebi notice": -0.5, "sebi ban": -0.6, "promoter selling": -0.5,
    "pledged shares": -0.4, "default": -0.5, "npa": -0.4,
    "insolvency": -0.6, "bankruptcy": -0.7, "debt restructuring": -0.4,
    # Market events
    "block deal": -0.3, "bulk deal sell": -0.3, "fii selling": -0.4,
    "margin call": -0.5, "circuit breaker": -0.4, "lower circuit": -0.5,
    # Results
    "loss widened": -0.5, "profit down": -0.4, "revenue miss": -0.4,
    "guidance cut": -0.5, "downgrade": -0.4, "rating cut": -0.4,
}

POSITIVE_BOOST = {
    # Corporate actions
    "buyback": +0.4, "bonus shares": +0.4, "dividend": +0.3,
    "stock split": +0.3, "promoter buying": +0.5, "fii buying": +0.4,
    # Results
    "profit up": +0.4, "revenue beat": +0.5, "record profit": +0.5,
    "strong results": +0.4, "upgrade": +0.4, "target raised": +0.4,
    # Orders / contracts
    "order win": +0.5, "new contract": +0.4, "expansion": +0.3,
    "joint venture": +0.3, "acquisition": +0.2,
}


class NewsMood(Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL  = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


@dataclass
class ArticleScore:
    headline:  str
    source:    str
    raw_vader: float     # VADER compound: -1 to +1
    boost:     float     # keyword adjustment
    final:     float     # raw_vader + boost, clamped to [-1, +1]
    keywords_found: list[str] = field(default_factory=list)


@dataclass
class NewsSentimentResult:
    symbol:          str
    mood:            NewsMood
    score:           float          # -1 to +1
    size_multiplier: float          # 0.0, 0.75, or 1.0
    allow_entry:     bool
    article_count:   int
    top_headlines:   list[str]
    articles:        list[ArticleScore] = field(default_factory=list)
    fetched_at:      datetime = field(default_factory=datetime.now)

    @property
    def is_stale(self) -> bool:
        """Re-fetch after 30 minutes."""
        return (datetime.now() - self.fetched_at) > timedelta(minutes=30)


# ── RSS Feed URLs ─────────────────────────────────────────────────

MARKET_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "https://feeds.reuters.com/reuters/INbusinessNews",
]

def symbol_feed(symbol: str) -> list[str]:
    """Google News RSS for a specific NSE symbol."""
    query = f"{symbol} NSE stock India"
    encoded = query.replace(" ", "+")
    return [
        f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
    ]


# ── Analyser ──────────────────────────────────────────────────────

class NewsSentimentAnalyser:

    CACHE_MINUTES = 30

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()
        self._cache: dict[str, NewsSentimentResult] = {}

    # ── Fetching ──────────────────────────────────────────────────

    def _fetch_headlines(self, urls: list[str], max_per_feed: int = 15) -> list[tuple[str, str]]:
        """Returns list of (headline, source_name)."""
        results = []
        for url in urls:
            try:
                feed = feedparser.parse(url)
                source = feed.feed.get("title", url.split("/")[2])
                for entry in feed.entries[:max_per_feed]:
                    title = entry.get("title", "").strip()
                    if title:
                        results.append((title, source))
            except Exception as exc:
                logger.debug(f"Feed fetch failed {url}: {exc}")
        return results

    # ── Scoring ───────────────────────────────────────────────────

    def _score_headline(self, headline: str, source: str) -> ArticleScore:
        text_lower = headline.lower()

        # VADER base score
        vs = self._vader.polarity_scores(headline)
        raw = vs["compound"]   # -1 to +1

        # Dampen VADER when headline is mostly event/reporting language
        dampen_count = sum(1 for w in DAMPEN_WORDS if w in text_lower)
        if dampen_count >= 2:
            raw = raw * 0.4   # reduce VADER weight on "awaited decision" style headlines

        # Keyword boosters
        boost = 0.0
        found = []
        for kw, adj in NEGATIVE_BOOST.items():
            if kw in text_lower:
                boost += adj
                found.append(kw)
        for kw, adj in POSITIVE_BOOST.items():
            if kw in text_lower:
                boost += adj
                found.append(kw)

        final = max(-1.0, min(1.0, raw + boost))

        return ArticleScore(
            headline=headline, source=source,
            raw_vader=round(raw, 3), boost=round(boost, 3),
            final=round(final, 3), keywords_found=found,
        )

    def _filter_relevant(
        self, articles: list[ArticleScore], symbol: str
    ) -> list[ArticleScore]:
        """Keep only articles that mention the symbol or are market-wide."""
        sym_lower = symbol.lower()
        relevant = []
        for a in articles:
            hl = a.headline.lower()
            # Include if headline mentions symbol, or is a market-wide piece
            if (sym_lower in hl or
                "nifty" in hl or "sensex" in hl or
                "market" in hl or "stock" in hl):
                relevant.append(a)
        return relevant if relevant else articles   # fallback: use all

    def _to_result(
        self, symbol: str, articles: list[ArticleScore]
    ) -> NewsSentimentResult:
        if not articles:
            return NewsSentimentResult(
                symbol=symbol, mood=NewsMood.NEUTRAL,
                score=0.0, size_multiplier=0.75, allow_entry=True,
                article_count=0, top_headlines=["No news found"],
            )

        avg_score = sum(a.final for a in articles) / len(articles)
        avg_score = round(avg_score, 4)

        if avg_score > 0.15:
            mood, mult, allow = NewsMood.POSITIVE, 1.00, True
        elif avg_score < -0.15:
            mood, mult, allow = NewsMood.NEGATIVE, 0.00, False
        else:
            mood, mult, allow = NewsMood.NEUTRAL, 0.75, True

        # Top 3 most extreme headlines (most negative or most positive)
        sorted_arts = sorted(articles, key=lambda a: abs(a.final), reverse=True)
        top = [a.headline[:80] for a in sorted_arts[:3]]

        return NewsSentimentResult(
            symbol=symbol, mood=mood, score=avg_score,
            size_multiplier=mult, allow_entry=allow,
            article_count=len(articles), top_headlines=top,
            articles=sorted_arts[:10],
        )

    # ── Public API ────────────────────────────────────────────────

    def analyse(self, symbol: str) -> NewsSentimentResult:
        """
        Analyse news sentiment for *symbol*.
        Returns cached result if < 30 min old.
        """
        cached = self._cache.get(symbol)
        if cached and not cached.is_stale:
            logger.debug(f"[NewsSentiment] {symbol} — using cached result")
            return cached

        logger.info(f"[NewsSentiment] Fetching news for {symbol}…")
        all_headlines: list[tuple[str, str]] = []

        # 1. Symbol-specific Google News
        all_headlines += self._fetch_headlines(symbol_feed(symbol), max_per_feed=10)

        # 2. General market feeds (shared across all symbols)
        if symbol not in self._cache:   # only fetch market feeds once per cycle
            all_headlines += self._fetch_headlines(MARKET_FEEDS, max_per_feed=8)

        # Score each headline
        articles = [self._score_headline(h, s) for h, s in all_headlines]

        # Filter to relevant ones
        articles = self._filter_relevant(articles, symbol)

        result = self._to_result(symbol, articles)
        self._cache[symbol] = result

        logger.info(
            f"[NewsSentiment] {symbol}  mood={result.mood.value}  "
            f"score={result.score:+.3f}  articles={result.article_count}  "
            f"allow={result.allow_entry}"
        )
        return result

    def analyse_batch(self, symbols: list[str]) -> dict[str, NewsSentimentResult]:
        """Analyse a list of symbols, respecting rate limits."""
        results = {}
        for i, sym in enumerate(symbols):
            results[sym] = self.analyse(sym)
            if i < len(symbols) - 1:
                time.sleep(0.5)   # be polite to RSS servers
        return results

    # ── Rich display ──────────────────────────────────────────────

    def print_summary(self, results: dict[str, NewsSentimentResult]):
        tbl = Table(
            title=f"📰 News Sentiment — {datetime.now().strftime('%d %b %Y  %H:%M')}",
            box=box.SIMPLE_HEAVY, title_style="bold cyan",
        )
        tbl.add_column("Symbol",    style="bold white", width=12)
        tbl.add_column("Mood",      width=10)
        tbl.add_column("Score",     justify="right", width=7)
        tbl.add_column("Articles",  justify="right", width=9)
        tbl.add_column("Entry",     width=6)
        tbl.add_column("Top Headline", width=55)

        mood_colors = {
            NewsMood.POSITIVE: "green",
            NewsMood.NEUTRAL:  "yellow",
            NewsMood.NEGATIVE: "red",
        }
        for sym, r in results.items():
            c = mood_colors[r.mood]
            tbl.add_row(
                sym,
                f"[{c}]{r.mood.value}[/]",
                f"[{c}]{r.score:+.3f}[/]",
                str(r.article_count),
                "✅" if r.allow_entry else "❌",
                r.top_headlines[0] if r.top_headlines else "—",
            )
        console.print(tbl)

"""What each ticker in the universe actually is, in plain language.

Reference data, not market data: it never changes at runtime and nothing
computes with it. It exists so the dashboard can say "NVDA · Nvidia" and one
line about what the company does, rather than assuming the reader already
knows all nineteen symbols by heart.

WHY THIS LIVES IN PYTHON AND NOT IN THE DASHBOARD
-------------------------------------------------
The universe is defined in `config/settings.yaml`. If this table lived in
TypeScript, adding a symbol there would silently render a blank name on the
page and nothing would complain. Here, `tests/test_instruments.py` asserts
every configured symbol has an entry, so adding a symbol without describing it
fails the build instead of shipping a gap.

Alpaca does supply a `name` field, but it reads "Apple Inc. Common Stock" and
"State Street SPDR S&P 500 ETF Trust". Accurate, and not what a person wants
on a dashboard row.

The `what` line is deliberately about the BUSINESS, not the stock. Nothing
here is a view on the price; the strategy does not read this file and must
not. Keeping opinions out of it is what stops a description quietly becoming
a thesis.
"""

from __future__ import annotations

INSTRUMENTS: dict[str, dict[str, str]] = {
    # -- index and broad market ------------------------------------------
    "SPY": {
        "name": "S&P 500 ETF",
        "what": "Tracks the 500 largest US listed companies. The default "
                "benchmark this strategy is measured against.",
    },
    "QQQ": {
        "name": "Nasdaq 100 ETF",
        "what": "Tracks the 100 largest non-financial Nasdaq companies. "
                "Heavily weighted toward big technology.",
    },
    # -- technology ------------------------------------------------------
    "AAPL": {
        "name": "Apple",
        "what": "Designs the iPhone, Mac and iPad, and sells services like "
                "the App Store and iCloud on top of that installed base.",
    },
    "MSFT": {
        "name": "Microsoft",
        "what": "Windows, Office and the Azure cloud platform. Azure and its "
                "OpenAI partnership are what tie it to the AI buildout.",
    },
    "META": {
        "name": "Meta Platforms",
        "what": "Facebook, Instagram and WhatsApp, funded almost entirely by "
                "advertising. Spending heavily on AI and data centres.",
    },
    "GOOGL": {
        "name": "Alphabet",
        "what": "Google Search, YouTube and Google Cloud, plus the Gemini AI "
                "models. Advertising is still most of the revenue.",
    },
    "PLTR": {
        "name": "Palantir",
        "what": "Data analysis software for governments, defence agencies and "
                "large companies. Long contracts, concentrated customers.",
    },
    # -- semiconductors --------------------------------------------------
    "NVDA": {
        "name": "Nvidia",
        "what": "Makes the GPUs that train and run most AI models, and the "
                "CUDA software nearly all of that work is written against.",
    },
    "AMD": {
        "name": "Advanced Micro Devices",
        "what": "CPUs and GPUs for PCs, servers and data centres. The main "
                "listed challenger to Nvidia in AI accelerators.",
    },
    "AVGO": {
        "name": "Broadcom",
        "what": "Networking silicon plus custom AI chips built to order for "
                "the largest cloud providers. Also owns VMware.",
    },
    "SMCI": {
        "name": "Super Micro Computer",
        "what": "Builds and assembles the server racks that data centres fill "
                "with AI chips, including liquid-cooled systems.",
    },
    # -- consumer and financial ------------------------------------------
    "AMZN": {
        "name": "Amazon",
        "what": "The largest Western online retailer, and AWS, the biggest "
                "cloud computing business. AWS earns most of the profit.",
    },
    "TSLA": {
        "name": "Tesla",
        "what": "Electric vehicles, battery storage and a self-driving "
                "programme. Deliveries and margins drive the numbers.",
    },
    "COIN": {
        "name": "Coinbase",
        "what": "A US cryptocurrency exchange earning fees on trading and "
                "custody. Revenue tracks crypto volumes closely.",
    },
    # -- defensive sleeves, added in variant 5 ---------------------------
    "TLT": {
        "name": "20+ Year Treasury ETF",
        "what": "Long-dated US government bonds. Tends to rise when rates "
                "fall, which is often when equities are falling.",
    },
    "IEF": {
        "name": "7-10 Year Treasury ETF",
        "what": "Medium-dated US government bonds. The same defensive idea as "
                "TLT but less sensitive to interest rate moves.",
    },
    "GLD": {
        "name": "Gold ETF",
        "what": "Holds physical gold bullion. Held here because it does not "
                "move with equities, not as a view on gold.",
    },
    "DBC": {
        "name": "Commodity Index Fund",
        "what": "A basket of energy, metals and agricultural futures. Gives "
                "the book something that can rise during inflation.",
    },
    "BIL": {
        "name": "1-3 Month T-Bill ETF",
        "what": "Very short US government debt, the closest thing to cash "
                "that still pays interest. Where the book sits out a storm.",
    },
}


def describe(symbol: str) -> dict[str, str]:
    """Name and one-line description, with a safe fallback.

    An unknown symbol returns the ticker as its own name rather than raising:
    a missing description should leave a dull dashboard row, never break the
    page. `tests/test_instruments.py` is what stops it going unnoticed.
    """
    return INSTRUMENTS.get(symbol, {"name": symbol, "what": ""})

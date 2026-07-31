"""Reliable literature search utilities powered by Tavily.

The module deliberately restricts results to scholarly publishers, indexing
services and preprint repositories. Search output is still retrieval-based and
must be critically evaluated by the user.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any
from urllib.parse import urlparse

TRUSTED_DOMAINS = [
    "pubs.acs.org",
    "pubs.rsc.org",
    "sciencedirect.com",
    "link.springer.com",
    "nature.com",
    "onlinelibrary.wiley.com",
    "academic.oup.com",
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "acs.figshare.com",
    "chemrxiv.org",
    "arxiv.org",
    "mdpi.com",
    "tandfonline.com",
    "journals.iucr.org",
    "science.org",
    "pnas.org",
    "cell.com",
]

# Temporary deployment key. Replace this single value when rotating the Tavily key.
TAVILY_API_KEY = "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"

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _doi(text: str) -> str | None:
    match = DOI_RE.search(text or "")
    return match.group(0).rstrip(".,;)") if match else None


def _api_key(explicit_key: str | None = None) -> str:
    """Return the app-bundled Tavily key; no user input or Streamlit secret is required."""
    key = (explicit_key or TAVILY_API_KEY).strip()
    if not key:
        raise RuntimeError("The bundled Tavily deployment key is empty.")
    return key


def search_literature(
    keyword: str,
    *,
    years_back: int = 5,
    max_results: int = 10,
    api_key: str | None = None,
    mof_focus: bool = True,
) -> list[dict[str, Any]]:
    """Search recent scholarly literature and return normalized results.

    Parameters
    ----------
    keyword:
        User topic, compound, material or process.
    years_back:
        Earliest publication/update date to request from Tavily.
    max_results:
        Number of results to request (5–20 recommended).
    api_key:
        Optional runtime override. If omitted, the bundled deployment key is used.
    mof_focus:
        Add MOF/materials-science context to reduce generic matches.
    """
    query = (keyword or "").strip()
    if not query:
        return []

    key = _api_key(api_key)
    try:
        from tavily import TavilyClient
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("The tavily-python package is not installed.") from exc

    context = " metal-organic framework MOF synthesis" if mof_focus else ""
    scholarly_query = f'{query}{context} research article journal DOI'
    start = date.today() - timedelta(days=365 * max(1, int(years_back)))

    client = TavilyClient(api_key=key)
    response = client.search(
        query=scholarly_query,
        search_depth="advanced",
        topic="general",
        max_results=max(5, min(int(max_results) * 2, 40)),
        include_answer=False,
        include_raw_content=False,
        include_domains=TRUSTED_DOMAINS,
        start_date=start.isoformat(),
    )

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in response.get("results", []):
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "Untitled result").strip()
        if not url or url in seen:
            continue
        domain = _domain(url)
        if not any(domain == d or domain.endswith("." + d) for d in TRUSTED_DOMAINS):
            continue
        seen.add(url)
        content = str(item.get("content") or "").strip()
        published = item.get("published_date") or item.get("date") or ""
        normalized.append(
            {
                "title": title,
                "url": url,
                "source": domain,
                "published_date": str(published),
                "score": float(item.get("score") or 0.0),
                "summary": content,
                "doi": _doi(" ".join([title, url, content])),
            }
        )

    # Prefer dated/recent records, while using Tavily relevance as a tie-breaker.
    normalized.sort(
        key=lambda r: (
            bool(r["published_date"]),
            r["published_date"],
            r["score"],
        ),
        reverse=True,
    )
    return normalized[: int(max_results)]

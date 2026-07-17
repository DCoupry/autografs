"""
Polite local fetching of external topology sources.

AuToGraFS does not bundle data whose licenses restrict redistribution
(the IZA zeolite structure database; EPINET, when it lands). Instead
``autografs-topologies`` fetches such sources *to the user's machine*
after showing the source's terms and getting an explicit acceptance —
interactively, or via ``--accept-licenses`` in scripts.

This module is the shared machinery: the acceptance gate, a resumable
on-disk cache (a re-run only downloads what is missing, so an
interrupted fetch continues where it stopped), an identifying
user agent, and a fixed politeness delay between requests.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from autografs import __version__
from autografs.data.iza_codes import IZA_CODES, cif_filename

logger = logging.getLogger(__name__)

USER_AGENT = (
    f"autografs-topologies/{__version__} "
    "(+https://github.com/DCoupry/autografs; polite bulk fetch)"
)

# seconds between consecutive downloads: a few requests per second at
# most, single connection
REQUEST_DELAY = 0.5

IZA_CIF_URL = "https://www.iza-structure.org/IZA-SC/cif/{filename}"


@dataclass(frozen=True)
class Source:
    """One external data source and the notice its use requires."""

    key: str
    title: str
    homepage: str
    notice: str


IZA_SOURCE = Source(
    key="iza",
    title="IZA-SC Database of Zeolite Structures",
    homepage="https://www.iza-structure.org/databases/",
    notice=(
        "The zeolite framework data about to be downloaded comes from\n"
        "the IZA-SC Database of Zeolite Structures (Ch. Baerlocher,\n"
        "L.B. McCusker and co-workers), copyright the Structure\n"
        "Commission of the International Zeolite Association.\n"
        "\n"
        "The files are fetched to YOUR machine for YOUR local use;\n"
        "AuToGraFS ships nothing derived from them. Use of the\n"
        "database is subject to its own terms - in particular,\n"
        "commercial use and redistribution need the Commission's\n"
        "consent. Please cite the database in published work and see\n"
        "https://www.iza-structure.org/databases/ for the full terms\n"
        "and the preferred citation."
    ),
)


def require_acceptance(source: Source, accept: bool = False) -> None:
    """Show a source's terms and require explicit acceptance.

    Parameters
    ----------
    source : Source
        The source about to be fetched.
    accept : bool, optional
        True (the ``--accept-licenses`` flag) records acceptance
        without prompting — for scripts and batch jobs.

    Raises
    ------
    SystemExit
        If the terms are declined, or if no interactive terminal is
        available to ask and ``accept`` was not passed.
    """
    banner = "=" * 66
    print(f"{banner}\n{source.title}\n{source.homepage}\n\n{source.notice}\n{banner}")
    if accept:
        logger.info(f"{source.title}: terms accepted via --accept-licenses.")
        return
    if not sys.stdin.isatty():
        raise SystemExit(
            f"Fetching from {source.title} needs the terms accepted; "
            "no terminal is available to ask - pass --accept-licenses "
            "to accept them non-interactively."
        )
    answer = input("Accept these terms and download? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        raise SystemExit("Terms declined; nothing downloaded.")


def default_cache_dir(key: str) -> Path:
    """The per-source on-disk cache location."""
    return Path.home() / ".autografs" / "cache" / key


def fetch_files(
    urls: dict[str, str],
    cache_dir: Path,
    delay: float = REQUEST_DELAY,
) -> dict[str, Path]:
    """Download a set of files into a resumable cache.

    Files already present (and non-empty) in ``cache_dir`` are not
    re-requested, so an interrupted run resumes. Downloads are
    sequential on one connection with ``delay`` seconds between
    requests, and each file is written atomically. Failures are
    logged and skipped, not raised.

    Parameters
    ----------
    urls : dict[str, str]
        Mapping of cache filename to URL.
    cache_dir : Path
        Where the files live.
    delay : float, optional
        Politeness delay between actual downloads (skipped files cost
        nothing).

    Returns
    -------
    dict[str, Path]
        Cache path per successfully available filename.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    available: dict[str, Path] = {}
    missing = {}
    for filename, url in sorted(urls.items()):
        target = cache_dir / filename
        if target.is_file() and target.stat().st_size > 0:
            available[filename] = target
        else:
            missing[filename] = url
    if not missing:
        logger.info(f"All {len(available)} files already cached in {cache_dir}.")
        return available
    logger.info(
        f"Fetching {len(missing)} files ({len(available)} already cached) "
        f"into {cache_dir}."
    )
    with requests.Session() as session:
        session.headers["User-Agent"] = USER_AGENT
        for filename, url in tqdm(sorted(missing.items()), unit="file"):
            target = cache_dir / filename
            try:
                response = session.get(url, timeout=60)
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.warning(f"Failed to fetch {url}: {exc}")
                continue
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(response.content)
            os.replace(tmp, target)
            available[filename] = target
            time.sleep(delay)
    return available


def fetch_iza_cifs(
    cache_dir: Path | None = None, accept_licenses: bool = False
) -> dict[str, Path]:
    """The IZA idealized framework CIFs, fetched to the local cache.

    Returns
    -------
    dict[str, Path]
        Cached CIF path per official framework code (prefixes like
        ``-CLO`` and ``*BEA`` keep their code; the filename on the
        server has them stripped).
    """
    require_acceptance(IZA_SOURCE, accept=accept_licenses)
    urls = {
        cif_filename(code): IZA_CIF_URL.format(filename=cif_filename(code))
        for code in IZA_CODES
    }
    cached = fetch_files(urls, cache_dir or default_cache_dir("iza"))
    return {
        code: cached[cif_filename(code)]
        for code in IZA_CODES
        if cif_filename(code) in cached
    }

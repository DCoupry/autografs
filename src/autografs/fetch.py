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


EPINET_SOURCE = Source(
    key="epinet",
    title="EPINET: Euclidean Patterns in Non-Euclidean Tilings",
    homepage="https://epinet.anu.edu.au",
    notice=(
        "The s-net geometry files about to be downloaded come from the\n"
        "EPINET database (S. Ramsden, V. Robins, S. Hyde and\n"
        "collaborators, hosted at the Australian National University).\n"
        "\n"
        "EPINET's content is licensed under Creative Commons\n"
        "Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND).\n"
        "That license does NOT permit redistribution of derived\n"
        "copies: the files are fetched to YOUR machine for YOUR own\n"
        "non-commercial use, and any topology library you convert them\n"
        "into must stay local - AuToGraFS ships nothing derived from\n"
        "EPINET, and neither should you. Published work should cite\n"
        "EPINET; see https://epinet.anu.edu.au for the terms and the\n"
        "preferred citation."
    ),
)

EPINET_CGD_URL = "https://epinet.anu.edu.au/snet_cgd_files/{name}.cgd"

# the catalogue holds ~14,532 s-nets with identifiers observed up to
# sqc14645 (numbering is sparse); the default sweep ceiling leaves a
# margin, and absent ids are negatively cached so re-runs skip them
EPINET_MAX_ID = 14700

# EPINET appears dormant (no visible maintenance since ~2019), so the
# full-catalogue fetch is extra polite: one file per second, a single
# connection, resumable
EPINET_REQUEST_DELAY = 1.0


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


def fetch_epinet_cgds(
    cache_dir: Path | None = None,
    accept_licenses: bool = False,
    max_id: int = EPINET_MAX_ID,
    delay: float = EPINET_REQUEST_DELAY,
) -> dict[str, Path]:
    """The EPINET s-net CGD files, fetched to the local cache.

    EPINET exposes no bulk endpoint - one ``.cgd`` per net page - so
    this sweeps ``sqc1..sqc<max_id>`` politely (one request per
    ``delay`` seconds, single connection, resumable). The numbering is
    sparse: an id that answers 404 is recorded in ``absent.txt`` in the
    cache directory and never re-requested, while transient failures
    (timeouts, 5xx) stay retryable on the next run. A full first fetch
    takes hours by design; interrupt and re-run freely.

    The cache is for LOCAL use only: EPINET's CC BY-NC-ND terms do not
    permit redistributing the files or anything converted from them
    (see ``EPINET_SOURCE`` and the acceptance gate).

    Returns
    -------
    dict[str, Path]
        Cached CGD path per s-net name (``sqc168`` -> ``.../sqc168.cgd``).
    """
    require_acceptance(EPINET_SOURCE, accept=accept_licenses)
    cache = Path(cache_dir) if cache_dir else default_cache_dir("epinet")
    cache.mkdir(parents=True, exist_ok=True)
    absent_file = cache / "absent.txt"
    absent: set[str] = set()
    if absent_file.is_file():
        absent = set(absent_file.read_text(encoding="utf-8").split())

    available: dict[str, Path] = {}
    todo: list[str] = []
    for index in range(1, max_id + 1):
        name = f"sqc{index}"
        target = cache / f"{name}.cgd"
        if target.is_file() and target.stat().st_size > 0:
            available[name] = target
        elif name not in absent:
            todo.append(name)
    if not todo:
        logger.info(
            f"All {len(available)} EPINET files already cached in {cache} "
            f"({len(absent)} ids known absent)."
        )
        return available
    logger.info(
        f"Fetching up to {len(todo)} EPINET files ({len(available)} cached, "
        f"{len(absent)} known absent) into {cache}; this is deliberately "
        f"slow ({delay:.1f} s/request) and resumable."
    )
    with requests.Session() as session:
        session.headers["User-Agent"] = USER_AGENT
        for name in tqdm(todo, unit="net"):
            target = cache / f"{name}.cgd"
            try:
                response = session.get(EPINET_CGD_URL.format(name=name), timeout=60)
            except requests.RequestException as exc:
                logger.warning(f"Failed to fetch {name}: {exc}")
                time.sleep(delay)
                continue
            if response.status_code == 404:
                absent.add(name)
                # persist immediately: absence knowledge is what makes
                # the sparse sweep resumable at all
                absent_file.write_text(
                    "\n".join(sorted(absent)) + "\n", encoding="utf-8"
                )
            elif response.ok and response.content.strip():
                tmp = target.with_suffix(target.suffix + ".tmp")
                tmp.write_bytes(response.content)
                os.replace(tmp, target)
                available[name] = target
            else:
                logger.warning(
                    f"Unexpected response for {name}: HTTP {response.status_code}"
                )
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

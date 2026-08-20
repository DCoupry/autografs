"""
Polite local fetching of external topology sources.

AuToGraFS does not bundle data whose licenses restrict redistribution
(the IZA zeolite structure database; the EPINET dataset release). Instead
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
import re
import sys
import time
import zipfile
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


def _is_cached(target: Path) -> bool:
    """A non-empty file counts as cached; a zero-byte one is a failed
    or interrupted write and is fetched again."""
    return target.is_file() and target.stat().st_size > 0


def _atomic_write(target: Path, data: bytes) -> None:
    """Write via a sibling .tmp then os.replace, so an interrupted run
    never leaves a half-written file that _is_cached would trust."""
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)


def _session() -> requests.Session:
    """A requests session carrying the identifying User-Agent."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


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
        "The s-net geometry files about to be downloaded are the EPINET\n"
        "project's dataset release on the ANU Open Research repository\n"
        "(V. Robins, S. Ramsden, S. Hyde, O. Delgado-Friedrichs:\n"
        "'Periodic net records from the EPINET database: sqc1 to\n"
        "sqc14645', doi:10.25911/hq20-mj54).\n"
        "\n"
        "The release is licensed under Creative Commons\n"
        "Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA). Your\n"
        "use of it - including any topology library this tool converts\n"
        "it into - must stay non-commercial and credit EPINET, and a\n"
        "shared derived copy must carry the same license. AuToGraFS\n"
        "itself ships nothing derived from EPINET. Published work\n"
        "should cite EPINET; see https://epinet.anu.edu.au and the DOI\n"
        "above for the terms and the preferred citation."
    ),
)

# the dataset release: one zip of 14,645 per-net .cgd files (sparse sqc
# numbering), replacing the old per-page sweep of epinet.anu.edu.au
EPINET_DATASET_DOI = "10.25911/hq20-mj54"
EPINET_ARCHIVE_NAME = "snet-cgd-files.zip"
EPINET_ARCHIVE_URL = (
    "https://datacommons.anu.edu.au/DataCommons/rest/records/anudc:6420/data/"
    + EPINET_ARCHIVE_NAME
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
        if _is_cached(target):
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
    with _session() as session:
        for filename, url in tqdm(sorted(missing.items()), unit="file"):
            target = cache_dir / filename
            try:
                response = session.get(url, timeout=60)
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.warning(f"Failed to fetch {url}: {exc}")
                continue
            _atomic_write(target, response.content)
            available[filename] = target
            time.sleep(delay)
    return available


def _extract_epinet_archive(archive: Path, cache: Path) -> dict[str, Path]:
    """Extract the release zip's ``sqc*.cgd`` members into the cache.

    Members are matched on their basename and written flat (which is
    also what makes the extraction zip-slip safe); files already
    present and non-empty are kept, so a cache partially populated by
    the pre-release per-page sweep is completed, not re-written. A
    corrupt archive (a truncated download) is deleted so the next run
    re-fetches it.
    """
    available: dict[str, Path] = {}
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in tqdm(bundle.namelist(), unit="net"):
                filename = member.rsplit("/", 1)[-1]
                if not re.fullmatch(r"sqc\d+\.cgd", filename):
                    continue
                target = cache / filename
                if not _is_cached(target):
                    _atomic_write(target, bundle.read(member))
                available[filename.removesuffix(".cgd")] = target
    except zipfile.BadZipFile:
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            f"The cached EPINET archive {archive} is corrupt (likely a "
            "truncated download); it has been removed - re-run to fetch "
            "it again."
        ) from None
    return available


def fetch_epinet_cgds(
    cache_dir: Path | None = None,
    accept_licenses: bool = False,
    max_id: int | None = None,
) -> dict[str, Path]:
    """The EPINET s-net CGD files, fetched to the local cache.

    Downloads the EPINET dataset release from the ANU Open Research
    repository (``EPINET_ARCHIVE_URL``, doi:10.25911/hq20-mj54) - one
    ~13 MB zip holding the full catalogue of per-net ``.cgd`` files -
    and extracts it into the cache. A cached archive is never
    re-downloaded, and files already extracted are kept, so re-runs
    cost nothing. This replaces the pre-release polite per-page sweep
    of epinet.anu.edu.au, which took hours.

    The release is CC BY-NC-SA: local use and anything derived from it
    must stay non-commercial and credit EPINET, and a shared derived
    copy must carry the same license (see ``EPINET_SOURCE`` and the
    acceptance gate). AuToGraFS bundles nothing derived from it.

    Parameters
    ----------
    max_id : int, optional
        Only return nets with sqc id up to this value - a subset
        filter for quick experiments (default: the full catalogue).

    Returns
    -------
    dict[str, Path]
        Cached CGD path per s-net name (``sqc168`` -> ``.../sqc168.cgd``).
    """
    require_acceptance(EPINET_SOURCE, accept=accept_licenses)
    cache = Path(cache_dir) if cache_dir else default_cache_dir("epinet")
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / EPINET_ARCHIVE_NAME
    if not _is_cached(archive):
        logger.info(
            f"Downloading the EPINET dataset release "
            f"(doi:{EPINET_DATASET_DOI}) into {cache}."
        )
        with _session() as session:
            response = session.get(EPINET_ARCHIVE_URL, timeout=600)
            response.raise_for_status()
        _atomic_write(archive, response.content)
    available = _extract_epinet_archive(archive, cache)
    logger.info(f"{len(available)} EPINET s-nets available in {cache}.")
    if max_id is not None:
        available = {
            name: path
            for name, path in available.items()
            if int(name.removeprefix("sqc")) <= max_id
        }
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

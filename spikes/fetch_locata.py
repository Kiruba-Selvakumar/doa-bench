"""Fetch the files for one LOCATA recording without downloading the whole archive.

LOCATA (Zenodo record 3630471, licence ODC-By 1.0) ships its development set as a single
6.2 GB dev.zip. Zenodo honours HTTP range requests, so the zip's central directory and
individual members can be read remotely; one recording on one array is ~30 MB.

    python spikes/fetch_locata.py                                  # task1/recording1/dicit
    python spikes/fetch_locata.py dev/task2/recording1/eigenmike/  # any member prefix

Files land in data/external/locata/, which is gitignored.
"""

import sys
import time
import zipfile
from pathlib import Path

from remote_zip import HTTPRangeFile

URL = "https://zenodo.org/records/3630471/files/dev.zip?download=1"
OUT = Path(__file__).resolve().parents[1] / "data" / "external" / "locata"


def main(prefix: str = "dev/task1/recording1/dicit/") -> None:
    t0 = time.perf_counter()
    remote = HTTPRangeFile(URL, block_size=1 << 22)
    archive = zipfile.ZipFile(remote)
    members = [m for m in archive.infolist() if m.filename.startswith(prefix) and not m.is_dir()]
    if not members:
        sys.exit(f"no members under {prefix!r}")
    for m in members:
        archive.extract(m, OUT)
    fetched = sum(len(b) for b in remote.cache.values()) / 1e6
    print(f"{len(members)} files -> {OUT / prefix}")
    print(
        f"{remote.requests} range requests, {fetched:.1f} MB fetched of "
        f"{remote.size / 1e9:.1f} GB, {time.perf_counter() - t0:.0f} s"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Read individual members of a remote zip over HTTP range requests.

Spike code, not part of the package. LOCATA's dev.zip is 6.2 GB; one recording is a few
tens of MB. zipfile only needs a seekable file object, so a range-request reader with a
block cache is enough to list the archive and extract single members.
"""

import io
import urllib.request


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str, block_size: int = 1 << 20):
        self.url, self.block_size, self.pos, self.cache = url, block_size, 0, {}
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
        self.requests = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _block(self, idx: int) -> bytes:
        if idx not in self.cache:
            start = idx * self.block_size
            end = min(start + self.block_size, self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req) as r:
                if r.status != 206:
                    raise OSError(f"server ignored range request (HTTP {r.status})")
                self.cache[idx] = r.read()
            self.requests += 1
        return self.cache[idx]

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            idx, off = divmod(self.pos + len(out), self.block_size)
            out += self._block(idx)[off : off + n - len(out)]
        self.pos += n
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

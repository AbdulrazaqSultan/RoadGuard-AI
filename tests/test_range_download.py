"""Keep malformed or truncated HTTP ranges out of the archive buffer."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from roadguard.data.range_download import _retrieve


class RangeClient:
    """Serve a controlled byte range without opening a network connection."""

    def __init__(self, header, chunks):
        self.header = header
        self.chunks = chunks

    @contextmanager
    def stream(self, method, url, *, params, headers):
        assert method == "GET" and headers["Range"] == "bytes=4-7"
        yield SimpleNamespace(
            status_code=206,
            headers={"Content-Range": self.header},
            iter_raw=lambda **_: iter(self.chunks),
        )


def test_valid_range_overwrites_only_its_slice(tmp_path):
    part = tmp_path / "data.part"
    part.write_bytes(b"000000000000")
    client = RangeClient("bytes 4-7/12", [b"ab", b"cd"])
    _retrieve(client, "https://example.com", part, 4, 7, 12)
    assert part.read_bytes() == b"0000abcd0000"


@pytest.mark.parametrize(
    ("header", "chunks", "error"),
    [
        ("bytes 0-3/12", [b"abcd"], "Unexpected Content-Range"),
        ("bytes 4-7/12", [b"abc"], "Incomplete archive range"),
        ("bytes 4-7/12", [b"abcde"], "Range exceeded"),
    ],
)
def test_bad_ranges_cannot_be_acknowledged(tmp_path, header, chunks, error):
    part = tmp_path / "data.part"
    part.write_bytes(b"0" * 12)
    with pytest.raises(OSError, match=error):
        _retrieve(RangeClient(header, chunks), "https://example.com", part, 4, 7, 12)

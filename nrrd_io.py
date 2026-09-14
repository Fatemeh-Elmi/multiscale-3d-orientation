"""Small self-contained NRRD reader for the manuscript's attached 3-D volumes.

Supports embedded RAW/GZIP NRRD files with scalar numeric data. Axis ordering is
preserved exactly as declared by the NRRD ``sizes`` field.
"""
from __future__ import annotations
import gzip
from pathlib import Path
import numpy as np

_DTYPE_MAP = {
    "unsigned char": np.uint8,
    "uchar": np.uint8,
    "uint8": np.uint8,
    "signed char": np.int8,
    "char": np.int8,
    "int8": np.int8,
    "short": np.int16,
    "short int": np.int16,
    "int16": np.int16,
    "ushort": np.uint16,
    "unsigned short": np.uint16,
    "uint16": np.uint16,
    "int": np.int32,
    "int32": np.int32,
    "uint": np.uint32,
    "uint32": np.uint32,
    "float": np.float32,
    "double": np.float64,
}


def read_nrrd(path: str | Path) -> tuple[np.ndarray, dict[str, str]]:
    path = Path(path)
    raw = path.read_bytes()
    sep = raw.find(b"\n\n")
    sep_len = 2
    if sep < 0:
        sep = raw.find(b"\r\n\r\n")
        sep_len = 4
    if sep < 0:
        raise ValueError(f"Could not locate NRRD header terminator in {path}")

    header_text = raw[:sep].decode("ascii", errors="strict")
    payload = raw[sep + sep_len :]
    fields: dict[str, str] = {}
    for line in header_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("NRRD"):
            continue
        if ":=" in line:
            key, value = line.split(":=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        fields[key.strip().lower()] = value.strip()

    dimension = int(fields.get("dimension", "0"))
    sizes = tuple(int(v) for v in fields["sizes"].split())
    if dimension != len(sizes):
        raise ValueError("NRRD dimension does not match sizes")
    if "data file" in fields:
        raise NotImplementedError("Detached NRRD data files are not supported by this compact reader")

    dtype_name = fields["type"].lower()
    if dtype_name not in _DTYPE_MAP:
        raise NotImplementedError(f"Unsupported NRRD type: {fields['type']}")
    dtype = np.dtype(_DTYPE_MAP[dtype_name])
    endian = fields.get("endian", "little").lower()
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<" if endian == "little" else ">")

    encoding = fields.get("encoding", "raw").lower()
    if encoding in {"gzip", "gz"}:
        payload = gzip.decompress(payload)
    elif encoding not in {"raw"}:
        raise NotImplementedError(f"Unsupported NRRD encoding: {encoding}")

    arr = np.frombuffer(payload, dtype=dtype)
    expected = int(np.prod(sizes))
    if arr.size != expected:
        raise ValueError(f"NRRD payload has {arr.size} values; expected {expected}")

    # NRRD axis 0 is fastest varying. Fortran reshape preserves that axis order
    # in a NumPy array whose shape exactly matches the NRRD sizes field.
    return arr.reshape(sizes, order="F"), fields

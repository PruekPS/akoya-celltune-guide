"""Lazy reading of multiplexed images: PhenoCycler-Fusion .qptiff, OME-TIFF, plain TIFF.

Whole-slide images are tens of GB, so nothing here loads a full image. The
file is opened through tifffile's zarr interface and every read is a slice:
one channel, one pyramid level, optionally one region.

Channel names come from, in order of preference:
  1. a sidecar text file (one name per line), when given
  2. the file's own metadata (QPTIFF per-page XML <Biomarker>/<Name>, or OME-XML)
  3. a placeholder ch00, ch01, ... (stage 00 then flags it)
Pixel size (µm per pixel at full resolution) comes from, in order:
  1. an explicit override
  2. OME-XML PhysicalSizeX, or QPTIFF XML ScanResolution / PixelSize
  3. the TIFF resolution tags, when their unit is centimetre or inch
"""
import pathlib
import re
import xml.etree.ElementTree as ET

import numpy as np
import tifffile
import zarr


class ImageHandle:
    def __init__(self, path, channel_names_file=None, pixel_size_um=None):
        self.path = pathlib.Path(path)
        self._tif = tifffile.TiffFile(self.path)
        series = self._tif.series[0]
        self.axes = series.axes
        self.dtype = np.dtype(series.dtype)
        self.format = ("qptiff" if self._tif.is_qpi else "ome-tiff" if self._tif.is_ome else "tiff")

        store = series.aszarr()
        z = zarr.open(store, mode="r")
        if isinstance(z, zarr.Group):
            keys = sorted(z.array_keys(), key=int)
            self._levels = [z[k] for k in keys]
        else:
            self._levels = [z]
        self._store = store

        # Normalise every level to (C, Y, X). Single-channel images get C=1.
        self._perm = self._axes_permutation(self.axes)
        self.n_channels = self.level_shape(0)[0]

        self.channel_names, self.channel_names_source = self._channel_names(channel_names_file)
        self.pixel_size_um, self.pixel_size_source = self._pixel_size(pixel_size_um)

    # -- geometry -------------------------------------------------------
    @staticmethod
    def _axes_permutation(axes):
        axes = axes.upper().replace("I", "C").replace("S", "C").replace("Q", "C")
        if axes in ("YX",):
            return ("add_c", None)
        if sorted(axes) != sorted("CYX"):
            raise ValueError(f"unsupported axes '{axes}': expected a 2D multichannel image (C, Y, X in any order)")
        return ("perm", tuple(axes.index(a) for a in "CYX"))

    @property
    def n_levels(self):
        return len(self._levels)

    def level_shape(self, level):
        shape = self._levels[level].shape
        kind, perm = self._perm
        if kind == "add_c":
            return (1,) + tuple(shape)
        return tuple(shape[i] for i in perm)

    def level_downsample(self, level):
        return self.level_shape(0)[2] / self.level_shape(level)[2]

    def pick_level(self, max_side):
        """Finest pyramid level whose longer side is <= max_side (coarsest if none is)."""
        for lvl in range(self.n_levels):
            _, h, w = self.level_shape(lvl)
            if max(h, w) <= max_side:
                return lvl
        return self.n_levels - 1

    # -- reading --------------------------------------------------------
    def read(self, channel, level=0, y=None, x=None):
        """Read one channel as a 2D array. y and x are (start, stop) at that level."""
        arr = self._levels[level]
        ys = slice(*y) if y else slice(None)
        xs = slice(*x) if x else slice(None)
        kind, perm = self._perm
        if kind == "add_c":
            if channel != 0:
                raise IndexError("single-channel image")
            return np.asarray(arr[ys, xs])
        index = [None, None, None]
        index[perm[0]], index[perm[1]], index[perm[2]] = channel, ys, xs
        return np.asarray(arr[tuple(index)])

    def chunk_rows(self, level=0):
        """Height in rows of one stored chunk (the whole image for untiled TIFFs)."""
        kind, perm = self._perm
        chunks = self._levels[level].chunks
        return chunks[0] if kind == "add_c" else chunks[perm[1]]

    def iter_bands(self, channel, band_height, level=0):
        """Yield (y0, y1, band) full-width horizontal bands of one channel.

        Reads each stored chunk once: for an untiled TIFF (one chunk per
        channel) the whole channel is read a single time and sliced; for a
        tiled .qptiff, bands are read chunk-row by chunk-row. Memory stays at
        one chunk-row (or one channel for untiled files).
        """
        _, h, _ = self.level_shape(level)
        step = max(band_height, self.chunk_rows(level))
        step = int(np.ceil(step / band_height) * band_height)
        for r0 in range(0, h, step):
            r1 = min(r0 + step, h)
            block = self.read(channel, level, y=(r0, r1))
            for y0 in range(r0, r1, band_height):
                y1 = min(y0 + band_height, r1)
                yield y0, y1, block[y0 - r0:y1 - r0]

    def read_downsampled(self, channel, max_side=2048):
        """A channel at roughly max_side pixels on its longer side, plus the factor used.

        Uses the pyramid when there is one; otherwise block-averages full
        resolution one horizontal strip at a time to bound memory.
        """
        lvl = self.pick_level(max_side)
        _, h, w = self.level_shape(lvl)
        if max(h, w) <= max_side or self.n_levels > 1:
            return self.read(channel, lvl).astype(np.float32), self.level_downsample(lvl)
        f = int(np.ceil(max(h, w) / max_side))
        out_h, out_w = h // f, w // f
        out = np.empty((out_h, out_w), np.float32)
        strip = max(f, (4096 // f) * f)
        for y0, y1, band in self.iter_bands(channel, strip, lvl):
            y1 = min(y1, out_h * f)
            if y1 <= y0:
                continue
            block = band[:y1 - y0, :out_w * f].astype(np.float32)
            out[y0 // f:y1 // f] = block.reshape((y1 - y0) // f, f, out_w, f).mean(axis=(1, 3))
        return out, float(f * self.level_downsample(lvl))

    def close(self):
        self._tif.close()

    # -- metadata -------------------------------------------------------
    def _channel_names(self, sidecar):
        if sidecar:
            names = [ln.strip() for ln in pathlib.Path(sidecar).read_text().splitlines() if ln.strip()]
            if len(names) != self.n_channels:
                raise ValueError(f"{sidecar} lists {len(names)} names but the image has "
                                 f"{self.n_channels} channels")
            return names, f"sidecar file {pathlib.Path(sidecar).name}"
        if self.format == "qptiff":
            names = self._qptiff_names()
            if names:
                return names, "QPTIFF page metadata"
        if self.format == "ome-tiff":
            names = self._ome_names()
            if names:
                return names, "OME-XML"
        return [f"ch{i:02d}" for i in range(self.n_channels)], "placeholder (no names found)"

    def _qptiff_pages(self):
        # The full-resolution channel pages come first in a QPTIFF, one per channel.
        return [p for p in self._tif.pages[:self.n_channels]]

    def _qptiff_names(self):
        names = []
        for page in self._qptiff_pages():
            try:
                root = ET.fromstring(page.description)
            except ET.ParseError:
                return None
            name = None
            for tag in ("Biomarker", "Name"):
                el = root.find(f".//{tag}")
                if el is not None and el.text and el.text.strip():
                    name = el.text.strip()
                    break
            names.append(name or "")
        return names if all(names) else None

    def _ome_names(self):
        try:
            root = ET.fromstring(self._tif.ome_metadata)
        except (ET.ParseError, TypeError):
            return None
        chans = [el.get("Name") for el in root.iter() if el.tag.endswith("Channel")]
        return chans if len(chans) == self.n_channels and all(chans) else None

    def _pixel_size(self, override):
        if override:
            return float(override), "explicit override"
        if self.format == "ome-tiff":
            try:
                root = ET.fromstring(self._tif.ome_metadata)
                for el in root.iter():
                    if el.tag.endswith("Pixels") and el.get("PhysicalSizeX"):
                        unit = el.get("PhysicalSizeXUnit", "µm")
                        val = float(el.get("PhysicalSizeX"))
                        if unit in ("µm", "um", "micron"):
                            return val, "OME-XML PhysicalSizeX"
                        if unit == "nm":
                            return val / 1000, "OME-XML PhysicalSizeX"
            except (ET.ParseError, TypeError, ValueError):
                pass
        if self.format == "qptiff":
            desc = self._tif.pages[0].description or ""
            m = re.search(r"<ScanResolution>\s*([\d.]+)\s*</ScanResolution>", desc) or \
                re.search(r"<PixelSize(?:Microns)?>\s*([\d.]+)\s*</PixelSize", desc)
            if m:
                return float(m.group(1)), "QPTIFF XML"
        page = self._tif.pages[0]
        tags = page.tags
        if "XResolution" in tags and "ResolutionUnit" in tags:
            num, den = tags["XResolution"].value
            unit = int(tags["ResolutionUnit"].value)
            if num and den:
                per_unit = num / den  # pixels per unit
                if unit == 3:  # centimetre
                    return 1e4 / per_unit, "TIFF resolution tags (cm)"
                if unit == 2 and per_unit not in (72, 96, 300):  # inch; 72/96/300 are screen/print defaults
                    return 25400 / per_unit, "TIFF resolution tags (inch)"
        return None, "not found"


def display_range(values, lo=0.5, hi=99.9):
    """Percentile window used for every image shown in reports and in QuPath.

    Applied identically to every channel and sample, so nobody tunes
    brightness by eye. Returns (low, high) as floats; high > low always.
    """
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(values, [lo, hi])
    if b <= a:
        b = a + 1.0
    return float(a), float(b)


def to_display(img, low, high, gamma=1.0):
    out = np.clip((img.astype(np.float32) - low) / (high - low), 0, 1)
    return out ** gamma if gamma != 1.0 else out

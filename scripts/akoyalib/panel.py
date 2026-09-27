"""Read and validate the two sheets every run is driven by: panel.csv and samples.csv.

Nothing about markers, species or sample layout is hardcoded in the stage
scripts; it all comes from these files.

panel.csv columns
-----------------
marker        name as it should appear in results (e.g. CD3e)
aliases       optional, ';'-separated other names the image metadata may use
              (e.g. "DAPI;DNA;Hoechst;DAPI-01")
role          nuclear | lineage | state | structural | exclude
segmentation  nuclear | membrane | (blank)  -- which composite image, if any,
              this channel contributes to in stage 02
status        ok | failed | unconfirmed  -- failed/unconfirmed markers stay
              visible in reports but are kept out of clustering and
              segmentation (e.g. a panel CD11c that did not stain)
notes         free text

samples.csv columns
-------------------
sample_id     short unique id, used for output folder names
image         path to the .qptiff / .ome.tif (relative to samples.csv or absolute)
species       human | mouse | canine
panel         path to that sample's panel.csv (relative to samples.csv or absolute)
condition     optional, any grouping column used later (stage 09)
he_image      optional, path to the matching H&E scan (stage 06)
channel_names optional, text file with one channel name per line, for images
              whose metadata carries no marker names (e.g. exported TIFFs)
pixel_size_um optional, µm per pixel at full resolution; overrides the file's
              metadata and is required when the file has none
exclusions    optional, QuPath GeoJSON of regions to leave out (folds, bubbles...)
"""
import pathlib

import pandas as pd

ROLES = {"nuclear", "lineage", "state", "structural", "exclude"}
SEGMENTATION = {"nuclear", "membrane", ""}
STATUS = {"ok", "failed", "unconfirmed"}
SPECIES = {"human", "mouse", "canine"}


class SheetError(ValueError):
    """A panel or samples sheet is malformed; the message says how to fix it."""


def _norm(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def read_panel(path):
    path = pathlib.Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"marker", "role"} - set(df.columns)
    if missing:
        raise SheetError(f"{path}: missing required column(s) {sorted(missing)}. "
                         f"Required: marker, role. Optional: aliases, segmentation, status, notes.")
    for col in ("aliases", "segmentation", "status", "notes"):
        if col not in df.columns:
            df[col] = ""
    df = df.apply(lambda s: s.str.strip())
    df["role"] = df["role"].str.lower()
    df["segmentation"] = df["segmentation"].str.lower()
    df["status"] = df["status"].str.lower().replace("", "ok")

    problems = []
    for i, row in df.iterrows():
        line = i + 2  # header is line 1
        if not row["marker"]:
            problems.append(f"line {line}: empty marker name")
        if row["role"] not in ROLES:
            problems.append(f"line {line} ({row['marker']}): role '{row['role']}' "
                            f"is not one of {sorted(ROLES)}")
        if row["segmentation"] not in SEGMENTATION:
            problems.append(f"line {line} ({row['marker']}): segmentation "
                            f"'{row['segmentation']}' must be nuclear, membrane or blank")
        if row["status"] not in STATUS:
            problems.append(f"line {line} ({row['marker']}): status '{row['status']}' "
                            f"is not one of {sorted(STATUS)}")
    dup = df["marker"][df["marker"].map(_norm).duplicated()]
    if len(dup):
        problems.append(f"duplicate marker names (case/punctuation-insensitive): {sorted(set(dup))}")
    if not (df["segmentation"] == "nuclear").any():
        problems.append("no channel has segmentation=nuclear; stage 02 needs one (usually DAPI)")
    if problems:
        raise SheetError(f"{path} has {len(problems)} problem(s):\n  - " + "\n  - ".join(problems))
    return df.reset_index(drop=True)


def read_samples(path):
    path = pathlib.Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"sample_id", "image", "species", "panel"} - set(df.columns)
    if missing:
        raise SheetError(f"{path}: missing required column(s) {sorted(missing)}. "
                         f"Required: sample_id, image, species, panel.")
    df = df.apply(lambda s: s.str.strip())
    base = path.parent
    problems = []
    if df.empty:
        raise SheetError(f"{path} has no sample rows")
    for col in ("channel_names", "pixel_size_um", "exclusions", "condition", "he_image"):
        if col not in df.columns:
            df[col] = ""
    for i, v in enumerate(df["pixel_size_um"]):
        if v:
            try:
                if not 0.05 <= float(v) <= 5:
                    problems.append(f"line {i + 2}: pixel_size_um {v} is outside 0.05-5 µm")
            except ValueError:
                problems.append(f"line {i + 2}: pixel_size_um '{v}' is not a number")
    for col in ("image", "panel", "he_image", "channel_names", "exclusions"):
        if col in df.columns:
            df[col] = [str((base / p).resolve()) if p and not pathlib.Path(p).is_absolute() else p
                       for p in df[col]]
    for i, row in df.iterrows():
        line = i + 2
        if row["species"].lower() not in SPECIES:
            problems.append(f"line {line} ({row['sample_id']}): species '{row['species']}' "
                            f"is not one of {sorted(SPECIES)}")
        for col in ("image", "panel", "channel_names", "exclusions"):
            if row[col] and not pathlib.Path(row[col]).exists():
                problems.append(f"line {line} ({row['sample_id']}): {col} not found: {row[col]}")
    if df["sample_id"].duplicated().any():
        problems.append(f"duplicate sample_id: {sorted(set(df['sample_id'][df['sample_id'].duplicated()]))}")
    bad_ids = [s for s in df["sample_id"] if not s or any(c in s for c in r' /\:*?"<>|')]
    if bad_ids:
        problems.append(f"sample_id must be non-empty with no spaces or path characters: {bad_ids}")
    if problems:
        raise SheetError(f"{path} has {len(problems)} problem(s):\n  - " + "\n  - ".join(problems))
    df["species"] = df["species"].str.lower()
    return df.reset_index(drop=True)


def match_channels(image_channel_names, panel):
    """Map each image channel to a panel row.

    Returns a DataFrame with one row per image channel: index, image_name,
    marker (or '' when unmatched), how (exact / alias / order / unmatched).
    Also returns the list of panel markers not found in the image.

    Matching is case- and punctuation-insensitive on marker name, then on
    aliases. If *no* image channel matches by name (typical when a reader
    only exposes filter names such as 'Opal 570'), falls back to panel order
    and says so -- the report then shows each channel image under its
    assigned name for a human to confirm.
    """
    lookup = {}
    for _, row in panel.iterrows():
        lookup.setdefault(_norm(row["marker"]), (row["marker"], "exact"))
        for alias in filter(None, (a.strip() for a in row["aliases"].split(";"))):
            lookup.setdefault(_norm(alias), (row["marker"], "alias"))

    rows = []
    for i, name in enumerate(image_channel_names):
        hit = lookup.get(_norm(name))
        rows.append({"index": i, "image_name": name,
                     "marker": hit[0] if hit else "", "how": hit[1] if hit else "unmatched"})
    table = pd.DataFrame(rows)

    if (table["how"] == "unmatched").all() and len(image_channel_names) == len(panel):
        table["marker"] = list(panel["marker"])
        table["how"] = "order"

    found = set(table["marker"])
    not_in_image = [m for m in panel["marker"] if m not in found]
    dup = table["marker"][(table["marker"] != "") & table["marker"].duplicated()]
    if len(dup):
        raise SheetError(f"several image channels map to the same panel marker(s) {sorted(set(dup))}; "
                         f"check aliases in the panel sheet")
    return table, not_in_image

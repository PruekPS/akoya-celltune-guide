"""Shared helpers for the numbered pipeline scripts (00_ingest.py, 01_image_qc.py, ...).

Scripts add their own directory to sys.path and import from here, so the
package works without installation:

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from akoyalib import imageio, panel, report
"""

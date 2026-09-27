"""Self-contained HTML reports, one per stage per sample.

Every report opens with the same header: what stage this is, its validation
status (the four labels used across this project), what the run was, and a
checklist of pass/warn/fail findings -- so the summary is read before the
detail. It closes with provenance: input checksums, every parameter, package
versions and the git commit, so any number in the report can be traced.

Reports embed their figures as PNG data, use only system fonts and make no
network requests, so they open offline (e.g. copied back from HiPerGator).
"""
import base64
import datetime as _dt
import html
import importlib.metadata
import io
import pathlib
import platform
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATUS_LABELS = ("designed, not built", "built, never run", "smoke-tested only", "validated on real data")

# Colours shared with docs/pipeline_blueprint.html: DAPI blue for the
# fluorescence track, eosin for H&E, hematoxylin violet for decisions.
PALETTE = {"if": "#2E57C9", "he": "#C0366C", "hema": "#5A3D94", "ink": "#1A1E29",
           "muted": "#5E6677", "ok": "#1D6E47", "warn": "#8F5400", "fail": "#A3212B"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 9.5, "axes.titlesize": 10,
    "axes.labelsize": 9.5, "axes.edgecolor": "#8A92A3", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": matplotlib.cycler(color=[PALETTE["if"], PALETTE["he"], PALETTE["hema"],
                                                "#2A8C8C", "#B07A12", "#6B7280"]),
})

_CSS = """
:root{--ground:#F5F6F8;--surface:#FFF;--ink:#1A1E29;--ink2:#3A4152;--muted:#5E6677;--line:#D8DCE4;
--wash:#ECEEF2;--if:#2E57C9;--hema:#5A3D94;--hema-wash:#F1EDF9;--ok:#1D6E47;--ok-wash:#E7F4EC;
--warn:#8F5400;--warn-wash:#FCF3E2;--fail:#A3212B;--fail-wash:#FBE9EA}
@media (prefers-color-scheme:dark){:root{--ground:#0E1117;--surface:#151A23;--ink:#E6E9F0;--ink2:#C3C9D6;
--muted:#98A1B3;--line:#2A3140;--wash:#1A202B;--if:#86A3FF;--hema:#BCA6F4;--hema-wash:#231B38;--ok:#6FD3A0;
--ok-wash:#11291D;--warn:#F0B35A;--warn-wash:#2A2112;--fail:#F28B91;--fail-wash:#2E1517}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font:16px/1.55 "Source Serif 4",Georgia,serif}
main{max-width:1080px;margin:0 auto;padding:40px 24px 80px;display:grid;gap:36px}
h1,h2,h3{font-family:"IBM Plex Sans Condensed","Arial Narrow","Helvetica Neue",Arial,sans-serif;margin:0;line-height:1.2}
h1{font-size:34px}h2{font-size:24px;padding-top:14px;border-top:2px solid var(--ink)}h3{font-size:18px}
p{margin:0;max-width:75ch}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.86em}
.eyebrow{font-family:ui-monospace,Menlo,monospace;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.stack{display:grid;gap:14px}
.meta{display:flex;flex-wrap:wrap;gap:8px 18px;font-family:"IBM Plex Sans Condensed",Arial,sans-serif;font-size:14.5px;color:var(--ink2)}
.pill{display:inline-block;font-family:ui-monospace,Menlo,monospace;font-size:12px;padding:2px 9px;border-radius:999px;
border:1px solid var(--hema);color:var(--hema);background:var(--hema-wash)}
.checks{display:grid;gap:0;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
.check{display:grid;grid-template-columns:58px 1fr;gap:12px;padding:10px 14px;border-bottom:1px solid var(--line);font-size:15px}
.check:last-child{border-bottom:none}
.tag{font-family:ui-monospace,Menlo,monospace;font-size:11px;font-weight:600;text-align:center;border-radius:3px;padding:2px 0;height:fit-content}
.tag.ok{color:var(--ok);background:var(--ok-wash)}.tag.warn{color:var(--warn);background:var(--warn-wash)}
.tag.fail{color:var(--fail);background:var(--fail-wash)}.tag.info{color:var(--muted);background:var(--wash)}
figure{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px}
figure img{display:block;max-width:100%;height:auto;margin:0 auto;background:#fff;border-radius:3px}
figcaption{font-size:14px;color:var(--muted);margin-top:8px;max-width:90ch}
.tbl{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 11px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-family:"IBM Plex Sans Condensed",Arial,sans-serif;font-size:13px;color:var(--muted);background:var(--ground)}
tr:last-child td{border-bottom:none}
.note{border-left:3px solid var(--warn);background:var(--warn-wash);padding:10px 14px;border-radius:0 4px 4px 0;font-size:15px;max-width:90ch}
.explain{border-left:3px solid var(--if);background:var(--surface);padding:10px 14px;border-radius:0 4px 4px 0;font-size:15px;max-width:90ch}
details{font-size:14px}summary{cursor:pointer;color:var(--ink2);font-family:"IBM Plex Sans Condensed",Arial,sans-serif}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
"""


def sha256sum(path, chunk=1 << 24):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def package_versions(names=("numpy", "scipy", "pandas", "scikit-image", "tifffile", "zarr",
                            "matplotlib", "tensorflow", "stardist", "csbdeep", "deepcell")):
    out = {"python": platform.python_version()}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def git_state(repo_dir):
    try:
        commit = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo_dir), "status", "--porcelain", "--", "scripts", "envs"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + (" (with uncommitted changes to scripts/envs)" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown (not a git checkout)"


def fig_to_img(fig, alt):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'


class Report:
    """Build one stage report.

    stage        e.g. "01"
    title        e.g. "Image QC"
    sample_id    the sample this report is about
    status       one of STATUS_LABELS: the *stage's* validation status
    run_note     what this particular run was (e.g. "public SPACEc demo data")
    """

    def __init__(self, stage, title, sample_id, status, run_note=""):
        if status not in STATUS_LABELS:
            raise ValueError(f"status must be one of {STATUS_LABELS}")
        self.stage, self.title, self.sample_id = stage, title, sample_id
        self.status, self.run_note = status, run_note
        self.checks = []
        self.blocks = []
        self.provenance = {}

    # -- summary checklist -------------------------------------------------
    def check(self, level, text):
        """level: ok | warn | fail | info. Text is plain language, one finding."""
        assert level in {"ok", "warn", "fail", "info"}
        self.checks.append((level, text))

    # -- body blocks --------------------------------------------------------
    def h2(self, text):
        self.blocks.append(f"<h2>{html.escape(text)}</h2>")

    def h3(self, text):
        self.blocks.append(f"<h3>{html.escape(text)}</h3>")

    def p(self, raw_html):
        self.blocks.append(f"<p>{raw_html}</p>")

    def explain(self, raw_html):
        """Blue-rule box: what this section measures and how to read it."""
        self.blocks.append(f'<div class="explain">{raw_html}</div>')

    def note(self, raw_html):
        self.blocks.append(f'<div class="note">{raw_html}</div>')

    def figure(self, fig, caption, alt=None):
        img = fig_to_img(fig, alt or caption)
        self.blocks.append(f"<figure>{img}<figcaption>{caption}</figcaption></figure>")

    def figures_side_by_side(self, items):
        """items: list of (fig, caption)."""
        inner = "".join(f"<figure>{fig_to_img(f, c)}<figcaption>{c}</figcaption></figure>" for f, c in items)
        self.blocks.append(f'<div class="grid2">{inner}</div>')

    def table(self, df, caption=None, max_rows=200, floatfmt="{:,.3g}"):
        shown = df.head(max_rows)
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in shown.columns)
        body = []
        for _, row in shown.iterrows():
            cells = []
            for v in row:
                if isinstance(v, float):
                    v = floatfmt.format(v)
                cells.append(f"<td>{html.escape(str(v))}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        more = f'<p class="mono" style="padding:8px 11px">… {len(df) - max_rows} more rows in the CSV output</p>' \
            if len(df) > max_rows else ""
        cap = f"<figcaption>{caption}</figcaption>" if caption else ""
        self.blocks.append(f'<div class="stack"><div class="tbl"><table><thead><tr>{head}</tr></thead>'
                           f'<tbody>{"".join(body)}</tbody></table>{more}</div>{cap}</div>')

    # -- provenance ---------------------------------------------------------
    def set_provenance(self, inputs=None, params=None, outputs=None, repo_dir=None):
        self.provenance = {
            "inputs": inputs or {}, "params": params or {}, "outputs": outputs or {},
            "versions": package_versions(), "git": git_state(repo_dir) if repo_dir else "unknown",
            "host": platform.node(), "when": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    def _provenance_html(self):
        pv = self.provenance
        if not pv:
            return ""

        def kv_table(d):
            rows = "".join(f"<tr><td class='mono'>{html.escape(str(k))}</td>"
                           f"<td class='mono'>{html.escape(str(v))}</td></tr>" for k, v in d.items())
            return f'<div class="tbl"><table><tbody>{rows}</tbody></table></div>'

        parts = ["<h2>Provenance</h2>",
                 f"<p class='meta'>Generated {html.escape(pv['when'])} on {html.escape(pv['host'])} · "
                 f"code at commit <span class='mono'>{html.escape(pv['git'])}</span></p>"]
        for label, key in (("Inputs", "inputs"), ("Parameters", "params"), ("Outputs", "outputs"),
                           ("Software versions", "versions")):
            if pv.get(key):
                parts.append(f"<details open><summary>{label}</summary>{kv_table(pv[key])}</details>")
        return "".join(parts)

    def write(self, path):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        order = {"fail": 0, "warn": 1, "info": 2, "ok": 3}
        checks = sorted(self.checks, key=lambda c: order[c[0]])
        n = {k: sum(1 for c in checks if c[0] == k) for k in order}
        check_html = "".join(f'<div class="check"><span class="tag {lvl}">{lvl.upper()}</span>'
                             f'<span>{text}</span></div>' for lvl, text in checks)
        summary = (f"{n['fail']} fail · {n['warn']} warn · {n['ok']} ok" if checks else "no automated checks")
        doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(self.stage)} {html.escape(self.title)} · {html.escape(self.sample_id)}</title>
<style>{_CSS}</style></head><body><main>
<header class="stack">
  <p class="eyebrow">Akoya analysis · stage {html.escape(self.stage)} · sample {html.escape(self.sample_id)}</p>
  <h1>{html.escape(self.title)}</h1>
  <div class="meta"><span>Stage status: <span class="pill">{html.escape(self.status)}</span></span>
  {'<span>Run: ' + html.escape(self.run_note) + '</span>' if self.run_note else ''}
  <span>Checks: {summary}</span></div>
</header>
<section class="stack"><h2>What needs attention</h2><div class="checks">{check_html or '<div class="check"><span class="tag info">INFO</span><span>No checks recorded.</span></div>'}</div></section>
{''.join(self.blocks)}
{self._provenance_html()}
</main></body></html>"""
        path.write_text(doc, encoding="utf-8")
        return path

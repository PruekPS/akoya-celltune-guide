# 11. Changing things

What to edit when something differs from the walkthrough. This is the page to
come back to.

---

## A different slide

Only `samples/` changes. Make a new file rather than editing the old one, so
you keep a record of what was run:

```bash
cp samples/my_slide.csv samples/second_slide.csv
nano samples/second_slide.csv
```

Change `sample_id` and `image`. Then run the same commands with the new
filename:

```bash
pipeline/sbatch_mail.sh --account=<GROUP> --qos=<GROUP> \
    pipeline/run_stages_00_01.slurm samples/second_slide.csv results

ACCOUNT=<GROUP> QOS=<GROUP> pipeline/run_stage_02_split.sh samples/second_slide.csv results
```

Because `sample_id` differs, results land in a different folder and nothing is
overwritten.

---

## A different output folder

The last argument. `results` is only a convention:

```bash
ACCOUNT=<GROUP> QOS=<GROUP> pipeline/run_stage_02_split.sh samples/my_slide.csv results_test
```

**Stages 00, 01 and 02 must all use the same output folder** for a given slide.
Stage 02 reads what stages 00 and 01 wrote, and looks for them under the folder
you give it. Mixing them up produces:

```
missing stage 00_ingest for <sample>
```

Separate output folders are useful for trying a setting without disturbing
results you want to keep — but then run **all three stages** into the new
folder, or copy `00_ingest/` and `01_image_qc/` across first.

---

## Only one slide from a multi-slide samples.csv

```bash
pipeline/sbatch_mail.sh --account=<GROUP> --qos=<GROUP> \
    pipeline/run_stage_02.slurm samples/all_slides.csv results --sample-id PS82
```

Repeat the flag for several: `--sample-id PS81 --sample-id PS82`.

---

## The project is somewhere else

Every command here is run **from the repository root** with relative paths like
`pipeline/...` and `samples/...`. If you put the repository elsewhere, only the
`cd` changes:

```bash
cd /path/to/wherever/you/put/akoya-pipeline
```

Then everything else is identical. Confirm first:

```bash
test -f scripts/02_segment.py && echo "OK, right folder" || echo "WRONG folder"
```

If you prefer, add a shortcut to `~/.bashrc` so it is one word:

```bash
echo "alias akoya='cd /blue/<GROUP>/<USER>/akoya-pipeline'" >> ~/.bashrc
source ~/.bashrc
```

Now typing `akoya` takes you there.

---

## Different cluster settings

Set these in front of the command, or `export` them once per session:

| Variable | Default | What it does |
|---|---|---|
| `ACCOUNT` | *(required)* | SLURM account |
| `QOS` | *(required)* | SLURM QoS |
| `GPU_PARTITION` | `hpg-rtx6000` | which GPU partition — **almost certainly different on your cluster** |
| `GPU_CPUS` / `GPU_TIME` / `GPU_MEM` | 8 / 08:00:00 / 100gb | GPU job resources |
| `CPU_CPUS` / `CPU_TIME` / `CPU_MEM` | 14 / 08:00:00 / 100gb | CPU job resources |
| `CELLSAM_BLOCK` | 512 | block size (page 6) |

Find your GPU partitions:

```bash
sinfo -o "%20P %30G" | grep -v "(null)"
```

There is often **no partition literally called `gpu`**, and the default
partition usually has none — you must name one. Example:

```bash
export ACCOUNT=myacct QOS=myqos GPU_PARTITION=gpu-a100
pipeline/run_stage_02_split.sh samples/my_slide.csv results
```

---

## Different segmentation settings

Anything after the output folder is passed straight to `02_segment.py`:

```bash
ACCOUNT=<GROUP> QOS=<GROUP> pipeline/run_stage_02_split.sh samples/my_slide.csv results \
    --cellsam-bbox-threshold 0.3 --min-nucleus-um2 8
```

Useful ones:

| Flag | Default | Effect |
|---|---|---|
| `--cellsam-bbox-threshold` | 0.4 | lower finds more cells, including more false ones |
| `--min-nucleus-um2` | 10 | drops nuclei smaller than this |
| `--expand-um` | 5 | how far the nucleus-expansion baseline grows |
| `--methods` | `cellsam,expansion` | which methods to run |
| `--celltune-method` | `cellsam` | which one gets exported to CellTune |

The full list:

```bash
python scripts/02_segment.py --help
```

---

## Turning the nested-label merge on

It is **off** by default, because it measured slightly worse than no merge
against hand labels and costs hours (page 6). To switch it on:

```bash
MERGE_NESTED=1 ACCOUNT=<GROUP> QOS=<GROUP> \
    pipeline/run_stage_02_split.sh samples/my_slide.csv results
```

Only worth doing if you have measured it on your own tissue and it helps there.
It is single-threaded and by far the slowest part of stage 02.

---

## Exporting nuclei instead of whole cells

```bash
CELLTUNE_METHOD=nuclei ACCOUNT=<GROUP> QOS=<GROUP> \
    pipeline/run_stage_02_split.sh samples/my_slide.csv results
```

The chosen method must be one that actually ran, or the export is skipped with
a warning.

---

Next: [11. When it goes wrong](11-troubleshooting.md)

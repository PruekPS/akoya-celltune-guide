# 12. When it goes wrong

Real errors, in the order people hit them.

---

## First: read the log

Every job writes one. Find it:

```bash
ls -lt logs/ | head
```

The newest is at the top. Read the end, where errors are:

```bash
tail -40 logs/<the-file>.out
```

And check how the job ended:

```bash
sacct -j <jobnumber> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

| State | Means |
|---|---|
| `COMPLETED` | finished normally |
| `FAILED` | the program returned an error |
| `TIMEOUT` | hit the `--time` limit |
| `OUT_OF_MEMORY` | needed more than `--mem` |
| `CANCELLED` | stopped — by you, or by a cluster policy |

---

## `command not found`

### `conda: command not found`
You did not load the module this session. Modules do not persist between
logins.
```bash
module load conda
```

### `python: command not found`
The environment is not active:
```bash
conda activate akoya-cellsam
```
This also happens in non-interactive SSH commands like
`ssh host 'python ...'`, where `module` may not exist at all. Use the full path
instead:
```bash
ssh host '/path/to/envs/akoya-cellsam/bin/python --version'
```

### `sbatch: error: Unable to open file pipeline/...`
You are not in the repository root.
```bash
cd /blue/<GROUP>/<USER>/akoya-pipeline
test -f scripts/02_segment.py && echo "OK" || echo "still wrong"
```

---

## `ModuleNotFoundError: No module named 'matplotlib'`

You ran the script with the wrong Python — usually your laptop's, or the
cluster's base Python. Two different mistakes look the same:

1. You forgot `conda activate akoya-cellsam`.
2. **You are running on your own computer instead of the cluster.** Check the
   path in the traceback: if it starts with `/Users/` or `C:\`, you are on your
   laptop. The pipeline runs on the cluster, through `sbatch`.

---

## Job dies in the first ten seconds

Almost always a path. The pre-flight check prints exactly which:

```
pre-flight:
  OK     samples sheet samples/my_slide.csv (1 rows)
  FAIL   image: /blue/g/u/Akoya_data/MySlide.qptiff does not exist
```

Fix the path in `samples/my_slide.csv` and resubmit. Page 4 has the checker
that catches this before you submit.

---

## `missing stage 00_ingest for <sample>`

Stage 02 cannot find stages 00/01. Either they have not run, or you gave a
different output folder. They must match — see page 11.

```bash
ls results/<your-sample-id>/
```

---

## `DEEPCELL_ACCESS_TOKEN is not set`

The token file is missing, empty, or misspelled. Check the length:

```bash
source ~/.config/deepcell_token && echo "${#DEEPCELL_ACCESS_TOKEN} characters"
```

`0` or an error means re-do page 3, step 3. Make sure it is one line beginning
`export DEEPCELL_ACCESS_TOKEN=`.

---

## Job cancelled for low GPU use

```
cancelled due to low GPU resource utilization
```

Your GPU sat idle too long. The usual cause is **the nested-label merge
running inside a single GPU job.** It is off by default; if you switched it on
(`MERGE_NESTED=1`), it runs for hours on one core after CellSAM has finished,
with the GPU doing nothing. Use `run_stage_02_split.sh` whenever the merge is
on (page 6).

The block size is not the cause. CellSAM ran a whole slide at the default
512 px — about 3,900 blocks — in a little over two hours on a GPU without
tripping the idle limit. Do not raise the block size to fix this; it costs
nearly half the cells (page 6).

**Check before redoing anything:**

```bash
cat results/<your-sample-id>/02_segment/cells_cellsam_raw.progress.json
```

If `"done": true`, segmentation finished and only the CPU part was lost.
Resume rather than restart — page 6, "Resuming".

---

## `TIMEOUT`

Raise `--time`. CellSAM is 14–16 h on CPU for a whole slide, about 2–3 h on a
GPU at the default 512 px block. If the job timed out after CellSAM had
finished, resume rather than restart, with the same block size as the original
run:

```bash
... pipeline/run_stage_02.slurm samples/my_slide.csv results --resume --cellsam-block-px 512
```

---

## `OUT_OF_MEMORY`

Raise `--mem`, within what your QoS allows:

```bash
CPU_MEM=200gb ACCOUNT=<GROUP> QOS=<GROUP> \
    pipeline/run_stage_02_split.sh samples/my_slide.csv results
```

Check your limit:

```bash
sacctmgr show qos <your-qos> format=Name,MaxTRES,MaxWall
```

---

## `Bus error (core dumped)` — exit code 135

The last line of the log looks like:

```
slurm_script: line 232: 476715 Bus error   (core dumped) python scripts/02_segment.py ...
```

and the job email says `FAILED (exit code 135)`. This is not a Python error: the
operating system stopped the program because a file it was using stopped being
readable or writable underneath it. Stage 02 keeps several slide-sized working
files on disk while it runs, so a brief storage hiccup on the cluster is the
usual cause. If other slides in the same batch finished, that is almost
certainly what happened.

**What to do:**

1. **Submit the same slide again**, once. Most of the time it simply works.
2. **If it fails again at the same point**, suspect the slide file itself —
   most often a transfer that did not finish. Compare its fingerprint on the
   cluster with the original:
   ```bash
   md5sum /blue/<GROUP>/<USER>/<data-folder>/<your-slide>.qptiff   # on the cluster
   md5 /path/on/your/laptop/<your-slide>.qptiff                     # on a Mac
   ```
   If they differ, transfer the slide again and rerun stages 00–01 and 02 for
   it.
3. **Delete the crash file.** `(core dumped)` leaves a file named `core.…` in
   the folder you submitted from. It is often several GB, counts against your
   storage, and nothing needs it:
   ```bash
   ls -lh core*
   rm core.<the full name it shows>
   ```

---

## Second job stuck at `(Dependency)`

Correct behaviour — it is waiting for the GPU job. If the first job **fails**,
the second never runs and shows `DependencyNeverSatisfied`. It will sit there
indefinitely, so cancel it and fix the first:

```bash
scancel <second-job-number>
```

This is worth checking on: a stalled chain sends no email until it is
cancelled, so it can wait unnoticed for a long time.

---

## No CellTune folder, but the job says COMPLETED

The export is logged as a **warning**, not an error.

```bash
grep -i celltune logs/akoya_02_cpu_*.out
```

Usually you asked to export a method that did not run — for example
`CELLTUNE_METHOD=mesmer` when only CellSAM ran.

---

## No emails

```bash
cat ~/.config/slurm_mail_user
```

Must be your **full address**, not a bare username. Check your spam folder. And
submit through `pipeline/sbatch_mail.sh`, not plain `sbatch` — on some clusters
the `#SBATCH --mail-type` line in a script is silently ignored, which is the
whole reason that wrapper exists.

---

## Still stuck

Collect these before asking for help — they are what anyone will ask for:

```bash
sacct -j <jobnumber> --format=JobID,State,Elapsed,MaxRSS,ExitCode
tail -40 logs/<the-log-file>.out
pwd
conda env list
```

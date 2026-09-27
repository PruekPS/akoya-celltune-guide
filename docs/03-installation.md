# 3. Installing the software (once)

You do this once. Later slides skip straight to page 4.

---

## Step 1 — load conda

Clusters keep software in **modules** you switch on as needed.

```bash
module load conda
```

Check it worked:

```bash
conda --version
```

A version number means yes. `command not found` means the module has a
different name on your cluster — try `module spider conda` to search.

---

## Step 2 — create the environment

From the repository root (check with `pwd` — see page 2):

```bash
conda env create -f environment-cellsam.yml -n akoya-cellsam
```

This downloads a few GB and takes **10–30 minutes**. Leave it running.

If it fails partway, remove and retry rather than resuming:

```bash
conda env remove -n akoya-cellsam
conda env create -f environment-cellsam.yml -n akoya-cellsam
```

> **On some clusters the login node forbids long installs.** If yours kills it,
> submit it instead:
> `srun --account=<GROUP> --qos=<GROUP> --cpus-per-task=4 --mem=16gb --time=02:00:00 --pty bash`
> then run the `conda env create` inside that session.

Activate it:

```bash
conda activate akoya-cellsam
```

Your prompt gains `(akoya-cellsam)`. Check the key pieces are there:

```bash
python -c "import torch, tifffile, stardist; print('ok', torch.__version__)"
```

---

## Step 3 — the DeepCell token

CellSAM downloads its trained weights from `users.deepcell.org`, which needs a
free token.

1. Go to **https://users.deepcell.org** and register.
2. Copy your access token.
3. Store it in a file only you can read:

```bash
mkdir -p ~/.config
nano ~/.config/deepcell_token
```

Put exactly one line in the file:

```
export DEEPCELL_ACCESS_TOKEN=paste-your-token-here
```

Save (`Ctrl`+`O`, Return) and exit (`Ctrl`+`X`). Then lock it down:

```bash
chmod 600 ~/.config/deepcell_token
```

**Two rules about the token.** It must be on **one line with no line break
inside the quotes** — a trailing newline makes an invalid HTTP header, and the
error message can print your token into a log file. And never paste it into a
script, a commit, or a chat window. The pipeline reads it from this file and
only ever prints its length.

Check it loads (this prints the length, not the token):

```bash
source ~/.config/deepcell_token && echo "token is ${#DEEPCELL_ACCESS_TOKEN} characters"
```

A number around 40 means good. `0` means the file is empty or misspelled.

---

## Step 4 — job emails

SLURM can email you when a job finishes. On some clusters the built-in setting
is ignored, so `pipeline/sbatch_mail.sh` sets it after submitting. Tell it your
address:

```bash
echo "you@your-university.edu" > ~/.config/slurm_mail_user
chmod 600 ~/.config/slurm_mail_user
```

Use your **full email address**, not just your username — a bare username often
silently delivers nowhere.

---

## Step 5 — check the panel file

`panels/` describes your antibodies. Look at the template:

```bash
head -3 panels/mouse_io_phenocode_template.csv
```

One row per marker in your panel. The columns are:

| Column | Required | Values | Meaning |
|---|---|---|---|
| `marker` | **yes** | e.g. `CD3e` | the name used in all results |
| `aliases` | no | `DAPI;DNA;Hoechst` | other names the image metadata might use, separated by `;` |
| `role` | **yes** | `nuclear`, `lineage`, `state`, `structural`, `exclude` | what kind of marker it is |
| `segmentation` | no | `nuclear`, `membrane`, or blank | which image it contributes to in stage 02 |
| `status` | no | `ok`, `failed`, `unconfirmed` | non-`ok` markers stay in reports but are kept out of segmentation |
| `notes` | no | free text | anything you want to remember |

`aliases` is how the pipeline copes with the instrument calling a channel
something slightly different from what you call it — list the variants and it
will match them.

**`segmentation` is the column that matters for this guide.** Exactly one
marker should be `nuclear` (usually DAPI). Every marker set to `membrane` is
averaged into the image CellSAM uses to find cell edges, so mark the ones that
outline cells — CD45, CD31, Pan-Cytokeratin, E-cadherin and similar — and leave
it blank for nuclear-only or purely functional markers.

Set `status` to `failed` for an antibody that did not stain. It then still
appears in the QC report, where you want to see it, but is kept out of the
membrane composite, where it would only add noise.

Copy the template and edit it for your panel:

```bash
cp panels/mouse_io_phenocode_template.csv panels/my_panel.csv
nano panels/my_panel.csv
```

---

## Done

You will not repeat this. From now on, each new slide starts at page 4. The one
thing to repeat **every time you log in** is:

```bash
module load conda
conda activate akoya-cellsam
```

---

Next: [4. Writing your samples.csv](04-samples-csv.md)

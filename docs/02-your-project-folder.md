# 2. Finding your way around

Almost every problem a beginner hits on a cluster is **being in the wrong
folder**. This page is about never being unsure.

---

## Three commands you will use constantly

| Command | Means | Answers |
|---|---|---|
| `pwd` | print working directory | "Where am I?" |
| `ls` | list | "What is here?" |
| `cd` | change directory | "Go there" |

Try them now:

```bash
pwd
```

```bash
ls
```

---

## Where to put the project

Home folders on clusters are small. Large data belongs on the big shared disk,
usually something like `/blue/<GROUP>/<USER>/`. Ask your support desk what
yours is.

Make a folder and go into it:

```bash
mkdir -p /blue/<GROUP>/<USER>
cd /blue/<GROUP>/<USER>
```

Then get this repository:

```bash
git clone https://github.com/PruekPS/akoya-celltune-guide.git akoya-pipeline
cd akoya-pipeline
```

`git clone` downloads it. The last word, `akoya-pipeline`, is the folder name it
creates — you may call it anything, but the rest of this guide assumes
`akoya-pipeline`.

---

## The check: am I in the right place?

Run this **every time** you are unsure. It is the single most useful habit here.

```bash
pwd && ls
```

You are in the right place if `pwd` ends in `akoya-pipeline` and `ls` shows:

```
docs  environment-cellsam.yml  panels  pipeline  README.md  samples  scripts
```

If you see something else, you are somewhere else. To get back:

```bash
cd /blue/<GROUP>/<USER>/akoya-pipeline
```

A stricter check — this prints `OK` only from the correct folder:

```bash
test -f scripts/02_segment.py && echo "OK, right folder" || echo "WRONG folder"
```

**This is what people mean by "the repository root".** It is the folder holding
`scripts/`, `pipeline/` and `README.md`. Every command in this guide is run
from there, not from inside `scripts/`.

---

## Reading a path

```
/blue/mygroup/jsmith/akoya-pipeline/scripts/02_segment.py
└──────────── the folders ─────────────┘ └──┘ └─────────┘
                                      folder   the file
```

- A path starting with `/` is **absolute** — it works from anywhere.
- A path not starting with `/` is **relative** — it is read from where you are
  now. `scripts/02_segment.py` only works from the repository root.
- `..` means "one folder up". `cd ..` goes up one level.
- `~` means your home folder.

---

## Where your slide files go

Keep raw images out of the repository. Make a sibling folder:

```bash
mkdir -p /blue/<GROUP>/<USER>/Akoya_data
```

So you end up with:

```
/blue/<GROUP>/<USER>/
├── akoya-pipeline/     <- this repository (code)
│   ├── scripts/
│   ├── pipeline/
│   ├── samples/
│   └── results/        <- created when you run things
└── Akoya_data/         <- your .qptiff files (data)
    └── MySlide.qptiff
```

Code and data separate. It means you can update the code without touching your
data, and your `.qptiff` files never get committed to git by accident.

---

## Getting a file onto the cluster

From a terminal **on your own computer** (not logged into the cluster):

```bash
scp /path/on/your/laptop/MySlide.qptiff <your-username>@<cluster-login-host>:/blue/<GROUP>/<USER>/Akoya_data/
```

A `.qptiff` is often 20–40 GB, so this takes a while. If your connection is
unreliable, `rsync` can resume:

```bash
rsync -avP /path/on/your/laptop/MySlide.qptiff <your-username>@<cluster-login-host>:/blue/<GROUP>/<USER>/Akoya_data/
```

If it stops, run exactly the same command again and it continues where it left
off.

Check it arrived at the right size:

```bash
ls -lh /blue/<GROUP>/<USER>/Akoya_data/
```

---

## Editing a file on the cluster

`nano` is the simplest editor:

```bash
nano samples/my_slide.csv
```

- Type normally; arrow keys move around.
- `Ctrl`+`O` then Return saves ("write Out").
- `Ctrl`+`X` exits.
- The `^` in nano's menu means `Ctrl`.

---

Next: [3. Installing the software](03-installation.md)

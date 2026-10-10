# 4. Writing your samples.csv

This is **the one file you write yourself**. It tells the pipeline which slide
to process and where everything is. One row per slide.

---

## Start from the example

```bash
cp samples/example.csv samples/my_slide.csv
nano samples/my_slide.csv
```

The example contains:

```csv
sample_id,image,species,panel,he_image,condition,notes
MySlide,/blue/GROUP/USER/Akoya_data/MySlide.qptiff,mouse,panels/my_panel.csv,,treated,"First test slide."
```

Keep the first line (the header) exactly as it is. Replace the second line.

---

## The columns

| Column | Required | What to put |
|---|---|---|
| `sample_id` | **yes** | A short name, no spaces. Becomes your results folder name. |
| `image` | **yes** | Full path to the `.qptiff`. |
| `species` | yes | `mouse` or `human`. |
| `panel` | **yes** | Path to your panel CSV. |
| `he_image` | no | An H&E scan of the same block, if you have one. Leave empty otherwise. |
| `condition` | no | Your experimental group. Free text. |
| `notes` | no | Anything you want to remember. |

### sample_id

Letters, numbers, underscores. **No spaces.** Results appear under
`results/<sample_id>/`, so `PS88` gives `results/PS88/`.

Good: `PS88`, `mouse_lung_01`, `Slide_A_treated`
Bad: `my slide` (space), `slide#1` (symbol)

### image — get this right

Use the **full path**, starting with `/`. Do not guess it. Print it:

```bash
ls /blue/<GROUP>/<USER>/Akoya_data/
```

Then build the path from what you see. Or have the shell give it to you exactly:

```bash
realpath /blue/<GROUP>/<USER>/Akoya_data/MySlide.qptiff
```

Copy that output into the column.

> **The mistake nearly everyone makes.** A path that works on your laptop does
> not work on the cluster. `/Users/you/Desktop/MySlide.qptiff` and
> `C:\Data\MySlide.qptiff` are laptop paths. On the cluster the file is
> somewhere under `/blue/...`. Always build the path **on the cluster**, with
> `ls` or `realpath`.

### panel

Path to the panel CSV you made on page 3. A relative path like
`panels/my_panel.csv` works because you run everything from the repository
root.

---

## Spaces and commas

CSV means "comma separated". If a value **contains a comma**, wrap it in double
quotes:

```csv
MySlide,/blue/g/u/MySlide.qptiff,mouse,panels/my_panel.csv,,treated,"Run 3, repeat of run 1"
```

Without the quotes the comma starts a new column and the row breaks.

**A path containing a space** also needs quotes:

```csv
MySlide,"/blue/g/u/My Data/MySlide.qptiff",mouse,panels/my_panel.csv,,,
```

Life is easier if you avoid spaces in folder names.

**Empty columns** are two commas with nothing between them. The `,,` in the
examples above is an empty `he_image`.

---

## Several slides in one file

Add one row per slide, same header:

```csv
sample_id,image,species,panel,he_image,condition,notes
PS81,/blue/g/u/Akoya_data/PS81.qptiff,mouse,panels/my_panel.csv,,control,
PS82,/blue/g/u/Akoya_data/PS82.qptiff,mouse,panels/my_panel.csv,,treated,
PS88,/blue/g/u/Akoya_data/PS88.qptiff,mouse,panels/my_panel.csv,,treated,
```

Every stage then processes all three in one job, one after another. Each gets
its own `results/<sample_id>/` folder.

To run only one row from a multi-slide file, add `--sample-id PS82` to the
command (see page 10).

---

## Check it before you submit

```bash
cat samples/my_slide.csv
```

Read it back. Then confirm the image path is real — this is the single most
common cause of a job dying in its first ten seconds:

```bash
cut -d, -f2 samples/my_slide.csv | tail -n +2 | tr -d '"' | while read f; do
  test -f "$f" && echo "FOUND:   $f" || echo "MISSING: $f"
done
```

Every line must say `FOUND`. If one says `MISSING`, fix the path before going on.

---

Next: [5. Stages 00 and 01](05-stage-00-01.md)

# 1. Getting on the cluster

Before anything else you need three things. Ask your PI or your cluster's
support desk for any you do not have.

| You need | What it looks like | Who gives it to you |
|---|---|---|
| A cluster account | a username, e.g. `jsmith` | your research computing centre |
| A SLURM **account** and **QoS** | two short names, often the same | your PI or support desk |
| Two-factor authentication set up | an app on your phone | your IT department |

Your SLURM account and QoS decide which computers you are allowed to use and
for how long. On many clusters this prints them:

```bash
sacctmgr show assoc user=$USER format=Account,QOS
```

Write both down. You will use them every time you submit work.

---

## Opening a terminal

- **macOS**: press `Cmd`+`Space`, type `Terminal`, press Return.
- **Windows**: open the Start menu, type `PowerShell`, press Return.
- **Linux**: `Ctrl`+`Alt`+`T` usually works.

A window appears with a line of text ending in `$` or `%`. That line is the
**prompt** — the computer saying it is ready. You type a command and press
Return. Throughout these pages, lines you type are shown in boxes like this:

```bash
echo hello
```

You type `echo hello` and press Return. You do **not** type the word `bash`.

---

## Logging in

```bash
ssh <your-username>@<cluster-login-host>
```

Replace `<your-username>` with your cluster username and `<cluster-login-host>`
with your cluster's address (your support desk will give you this; it looks
like `hpg.rc.university.edu`).

The first time, it asks:

```
The authenticity of host ... can't be established.
Are you sure you want to continue connecting (yes/no)?
```

Type `yes` and press Return. This happens once.

Then it asks for your password. **Nothing appears as you type** — no dots, no
stars. That is deliberate, not a broken keyboard. Type it and press Return.

Then two-factor: approve the push on your phone, or type the code.

You are in when the prompt changes to something like:

```
[jsmith@login3 ~]$
```

That tells you three things: who you are (`jsmith`), which machine you are on
(`login3`), and where you are (`~`, meaning your home folder).

### Logging out

```bash
exit
```

---

## The one rule about login nodes

The machine you land on is a **login node**. It is shared by everyone. It is
for editing files and submitting work — **never for running the pipeline**.

Running heavy work on a login node slows the cluster for everyone and your
account may be suspended. Every command in this guide that does real work is
submitted with `sbatch`, which sends it to a **compute node** instead. If you
ever find yourself typing `python scripts/02_segment.py ...` directly at the
prompt, stop: that is the wrong way, and page 6 shows the right one.

---

## If the connection drops

Cluster sessions time out, and Wi-Fi drops. If your terminal freezes or says
`Broken pipe`, close it, open a new one, and `ssh` in again. **Work you have
already submitted with `sbatch` keeps running** — it is not tied to your
session. That is the main reason everything here is submitted rather than run
directly.

---

Next: [2. Finding your way around](02-your-project-folder.md)

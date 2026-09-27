#!/bin/bash
# Submit a job and make sure it will email the account's address.
#
#   J=$(pipeline/sbatch_mail.sh --account=<ACCOUNT> --qos=<QOS> [options] pipeline/x.slurm args...)
#
# Takes exactly what sbatch takes and prints only the job id, so it chains:
#   J2=$(pipeline/sbatch_mail.sh --dependency=afterok:$J ...)
#
# Why it exists: every pipeline/*.slurm carries #SBATCH --mail-type, but on
# HiPerGator that directive did not reach a job (42119732 showed no MailType at
# all, while its other #SBATCH lines applied). Setting it with scontrol after
# submission has worked on every job, running or pending, so this does that and
# then checks.
#
# The address: the account default (bare username, no --mail-user) delivered
# nothing, so a full address is set. It is read from ~/.config/slurm_mail_user
# on the cluster, one line, mode 600 -- never from this repository.
set -euo pipefail
MAIL_TYPE="${MAIL_TYPE:-END,FAIL,INVALID_DEPEND}"
MAIL_USER="${SLURM_MAIL_USER:-$(head -1 "$HOME/.config/slurm_mail_user" 2>/dev/null || true)}"
[ -n "$MAIL_USER" ] || echo "WARNING: no ~/.config/slurm_mail_user -- falling back to the account default, which has not delivered" >&2

out=$(sbatch --parsable "$@")
j="${out%%;*}"                       # --parsable prints id or id;cluster
case "$j" in ''|*[!0-9]*) echo "sbatch did not return a job id: $out" >&2; exit 1 ;; esac

scontrol update JobId="$j" MailType="$MAIL_TYPE" ${MAIL_USER:+MailUser="$MAIL_USER"} >&2 || true
if scontrol show job "$j" | tr ' ' '\n' | grep -q '^MailType=.*END'; then
  echo "job $j: email on $MAIL_TYPE${MAIL_USER:+ to the address in ~/.config/slurm_mail_user}" >&2
else
  echo "WARNING: job $j is queued but carries no MailType -- it will not email" >&2
fi
echo "$j"

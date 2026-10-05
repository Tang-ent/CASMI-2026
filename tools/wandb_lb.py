"""Record a competition submission's leaderboard score as a wandb run (job_type "lb").

    python tools/wandb_lb.py --group M2 [--ref 56821350] [--eval-run <k08 run id>] [--note "gate:0.9"]
Defaults to the latest submission. The run is named after the submitting kernel and version; a second call
for the same submission updates the existing run. --eval-run also writes lb_public into that run's summary.
"""
import argparse
import os
import re
import sys

import wandb
from kaggle.api.kaggle_api_extended import KaggleApi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from casmi26.track import ENTITY, PROJECT  # noqa: E402

COMP = "enveda-CASMI26-molecule-id-mass-spectra"


def score(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, help="experiment group, e.g. M2")
    ap.add_argument("--ref", type=int, default=None, help="submission ref (default: latest)")
    ap.add_argument("--eval-run", default=None, help="wandb run id of the k08 eval that chose the submitted score")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    api = KaggleApi()
    api.authenticate()
    subs = api.competition_submissions(COMP)
    s = next(x for x in subs if x.ref == a.ref) if a.ref else subs[0]
    m = re.search(r"/code/[^/]+/([^?]+)\?scriptVersionId=(\d+)", s.url or "")
    kernel, version = (m.group(1), int(m.group(2))) if m else ("", None)
    cfg = dict(submission_ref=s.ref, date=str(s.date), kernel=kernel, script_version_id=version,
               description=s.description, status=str(s.status), eval_run=a.eval_run, note=a.note)
    summ = dict(lb_public=score(s.public_score), lb_private=score(s.private_score))
    print(cfg, summ)

    wapi = wandb.Api()
    path = f"{ENTITY}/{PROJECT}"
    try:
        old = list(wapi.runs(path, filters={"config.submission_ref": s.ref}))
    except (ValueError, wandb.errors.CommError):  # プロジェクトがまだ無い
        old = []
    if old:
        old[0].summary.update(summ)
        print("updated", old[0].url)
    else:
        run = wandb.init(entity=ENTITY, project=PROJECT, job_type="lb", group=a.group, name=f"lb-{kernel}-v{version}",
                         config=cfg, settings=wandb.Settings(console="off"))
        run.summary.update(summ)
        print("created", run.url)
        run.finish()
    if a.eval_run:
        er = wapi.run(f"{path}/{a.eval_run}")
        er.summary.update(dict(lb_public=summ["lb_public"], lb_submission_ref=s.ref))
        print("eval run updated", er.url)


if __name__ == "__main__":
    main()

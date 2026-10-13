"""Record a competition submission's leaderboard score as a wandb run (job_type "lb").

    python tools/wandb_lb.py --group M2 [--ref 56821350] [--eval-run <k08 run id>] [--note "gate:0.9"]
    python tools/wandb_lb.py --group M7 --variant-json out/m7_k36/v3/variant.json   # M7: k36 submissions
Defaults to the latest submission. --variant-json (k36 output) adds the variant name and its local eval numbers to the
run and appends the point to tools/lb_points.csv (once per submission ref) for tools/lb_fit.py. The run is named after the submitting kernel and version; a second call
for the same submission updates the existing run. --eval-run also writes lb_public into that run's summary.
"""
import argparse
import csv
import json
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
    ap.add_argument("--variant-json", default=None, help="k36 variant.json of the submitted version")
    ap.add_argument("--points", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lb_points.csv"))
    a = ap.parse_args()
    vj = json.load(open(a.variant_json)) if a.variant_json else None
    api = KaggleApi()
    api.authenticate()
    subs = api.competition_submissions(COMP)
    s = next(x for x in subs if x.ref == a.ref) if a.ref else subs[0]
    m = re.search(r"/code/[^/]+/([^?]+)\?scriptVersionId=(\d+)", s.url or "")
    kernel, version = (m.group(1), int(m.group(2))) if m else ("", None)
    cfg = dict(submission_ref=s.ref, date=str(s.date), kernel=kernel, script_version_id=version,
               description=s.description, status=str(s.status), eval_run=a.eval_run, note=a.note)
    summ = dict(lb_public=score(s.public_score), lb_private=score(s.private_score))
    if vj:
        cfg.update(variant=vj["variant"], src_sha=vj.get("src_sha"), eval=vj.get("eval"),
                   **{f"local_{k}": v for k, v in vj.get("local", {}).items()})
    print(cfg, summ)
    if vj and summ["lb_public"] is not None:
        with open(a.points) as fh:
            known = {r["submission_ref"] for r in csv.DictReader(fh)}
        if str(s.ref) not in known:
            loc = vj.get("local", {})
            row = [str(s.date)[:10], f"{kernel} v{version}", vj["variant"], s.ref, summ["lb_public"]] + \
                  [loc.get(k, "") for k in ("A", "B_A", "B_N", "G_N")] + [f"{vj.get('eval')} {loc.get('eval_model', '')}"]
            with open(a.points, "a", newline="") as fh:
                csv.writer(fh).writerow(row)
            print("appended to", a.points)

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

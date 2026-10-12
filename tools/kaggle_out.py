"""Fetch a kernel's log (and optionally output files matching a regex) without downloading everything.

    python tools/kaggle_out.py <user>/casmi26-k02-features -p out/k02 --files 'results.csv|query_info.csv'
The log is saved as <slug>.log and its printed text is extracted to <slug>.txt.
"""
import argparse
import json
import os
import re

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def list_output(kernel):
    """Output listing of the kernel's latest version: .files (file_name, url) and .log."""
    owner, slug = kernel.split("/")
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as k:
        req = ApiListKernelSessionOutputRequest()
        req.user_name, req.kernel_slug = owner, slug
        return k.kernels.kernels_api_client.list_kernel_session_output(req)


def download(f, path):
    out = os.path.join(path, f.file_name)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with requests.get(f.url, stream=True) as r:  # streamed: the cache files are several GB
        r.raise_for_status()
        with open(out, "wb") as fh:
            for chunk in r.iter_content(1 << 23):
                fh.write(chunk)
    print("downloaded", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("-p", "--path", default=".")
    ap.add_argument("--files", default=None, help="regex of output file names to download")
    a = ap.parse_args()
    slug = a.kernel.split("/")[1]
    resp = list_output(a.kernel)
    os.makedirs(a.path, exist_ok=True)
    print("output files:", ", ".join(f.file_name for f in resp.files))
    if a.files:
        for f in resp.files:
            if re.search(a.files, f.file_name):
                download(f, a.path)
    if resp.log:
        open(os.path.join(a.path, slug + ".log"), "w").write(resp.log)
        try:
            text = "".join(e.get("data", "") for e in json.loads(resp.log))
        except Exception:
            text = resp.log
        open(os.path.join(a.path, slug + ".txt"), "w").write(text)
        print(text[-6000:])


if __name__ == "__main__":
    main()

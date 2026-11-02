"""Official AI Hub shell wrapper. Keys stay in the child environment, never argv/logs."""
from pathlib import Path
import hashlib
import json
import os
import re
from urllib.parse import urlparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile

SHELL_URL = "https://api.aihub.or.kr/api/aihubshell.do"
GUIDE_URL = "https://aihub.or.kr/devsport/apishell/list.do"
DATASET_KEY = "189"


def proxy_environment():
    """Only AI Hub child commands use the optional, local Tailscale proxy."""
    env = dict(os.environ)
    proxy = env.get('AIHUB_PROXY_URL', '')
    if proxy:
        parsed = urlparse(proxy)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or not parsed.port or parsed.username or parsed.password:
            raise ValueError('AIHUB_PROXY_URL must be a local HTTP proxy without credentials')
        for name in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            env[name] = proxy
        for name in ['ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy']:
            env[name] = ''
    return env


def aihub_opener():
    env = proxy_environment()
    if env.get('AIHUB_PROXY_URL'):
        return urllib.request.build_opener(urllib.request.ProxyHandler({
            'http': env['AIHUB_PROXY_URL'], 'https': env['AIHUB_PROXY_URL']}))
    return urllib.request.build_opener()


def install_shell(cache):
    if sys.platform != "linux":
        raise RuntimeError("aihubshell requires Linux; run this cell in Google Colab.")
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    tool = cache / "aihubshell"
    if not tool.exists():
        with aihub_opener().open(SHELL_URL, timeout=60) as response:
            payload = response.read(1024 * 1024)
        if not payload.startswith(b"#!/bin/bash"):
            raise RuntimeError("Official shell response is not a bash script.")
        pending = tool.with_suffix(".tmp"); pending.write_bytes(payload)
        subprocess.run(["bash", "-n", str(pending)], check=True, capture_output=True)
        pending.replace(tool)
    (cache / "tool_source.json").write_text(json.dumps({
        "url": SHELL_URL, "sha256": hashlib.sha256(tool.read_bytes()).hexdigest()
    }, indent=2), encoding="utf-8")
    return tool.resolve()


def parse_catalog(text):
    rows = []; keys = set(); parents = {}
    for line in text.splitlines():
        match = re.match(r"^(.*?)([^│├└─\s].*?)\s*\|\s*([0-9.,]+)\s*([KMGT]?B)\s*\|\s*(\d+)\s*$", line, re.I)
        if match:
            prefix, name, size, unit, key = match.groups()
            if key in keys: raise ValueError("Duplicate file key in catalog")
            keys.add(key)
            depth = len(prefix.expandtabs(4))
            parent = "/".join(parents[k] for k in sorted(parents) if k < depth)
            rows.append({"file_key": key, "name": name.strip(), "path": (parent + "/" + name.strip()).lstrip("/"),
                         "size_bytes": int(float(size.replace(",", "")) * 1024 ** {"B":0,"KB":1,"MB":2,"GB":3,"TB":4}[unit.upper()])})
        elif line.strip() and not "|" in line:
            branch = re.match(r"^([│ ├└─]*)([^│├└─ ].*)$", line)
            if branch and any(c in branch[1] for c in "├└"):
                depth = len(branch[1]); parents = {k:v for k,v in parents.items() if k < depth}
                parents[depth] = branch[2].strip()
    return rows


def list_files(cache, dataset_key=DATASET_KEY):
    if not str(dataset_key).isdigit(): raise ValueError("Numeric dataset key required")
    tool = install_shell(cache)
    result = subprocess.run(["bash", str(tool), "-mode", "l", "-datasetkey", str(dataset_key)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, env=proxy_environment())
    rows = parse_catalog(result.stdout)
    if result.returncode or not rows:
        raise RuntimeError("AI Hub file listing failed or returned an unknown format. Retry later; check the official dataset key and API service.")
    Path(cache, "catalog_" + str(dataset_key) + ".json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def selection(rows, file_keys):
    keys = [str(k).strip() for k in file_keys]
    if not keys or len(set(keys)) != len(keys) or any(not k.isdigit() for k in keys):
        raise ValueError("Select explicit, unique numeric file keys; whole-dataset downloads are disabled.")
    available = {r["file_key"]: r for r in rows}
    if any(k not in available for k in keys): raise ValueError("Selected file key is absent from the current catalog")
    return [available[k] for k in keys]


def check_space(target, rows):
    required = sum(r["size_bytes"] for r in rows) * 3
    if not required: raise ValueError("Cannot estimate download size")
    if shutil.disk_usage(target).free < required:
        raise ValueError(f"Insufficient disk space: at least {required / 2**30:.2f} GiB required (3x archive size).")
    return required


def download_files(cache, destination, file_keys, api_key, dataset_key=DATASET_KEY):
    if not api_key or not api_key.strip(): raise ValueError("AI Hub API key required via Colab Secrets or hidden input")
    rows = selection(list_files(cache, dataset_key), file_keys)
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    check_space(destination, rows)
    # A new directory prevents the shell from merging/deleting unrelated existing archives.
    job = destination / ("files_" + "_".join(sorted(r["file_key"] for r in rows)))
    if job.exists(): raise FileExistsError("Download directory already exists. Inspect it and choose a new destination before retrying.")
    job.mkdir()
    env = proxy_environment(); env["AIHUB_APIKEY"] = api_key.strip()
    command = ["bash", str(install_shell(cache)), "-mode", "d", "-datasetkey", str(dataset_key),
               "-filekey", ",".join(r["file_key"] for r in rows)]
    # Shell can return zero even on HTTP failures. Require its success marker AND complete archives.
    print("AI Hub download started. Progress is recorded in", job / "download.log", flush=True)
    success = False; failed = False
    with (job / "download.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=job, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                safe = line.replace(api_key.strip(), "[REDACTED]")
                log.write(safe); log.flush()
                if "Request successful with HTTP status 200" in safe: success = True
                if any(token in safe for token in ("Download failed", "Error msg:", "tar: ", "Cannot open")): failed = True
                # Keep archive filenames and curl progress in the local log; no raw API error body in notebook output.
            code = process.wait()
        except BaseException:
            process.terminate(); process.wait(); raise
        finally:
            env.pop("AIHUB_APIKEY", None)
    if code or failed or not success:
        raise RuntimeError("AI Hub download failed. Check API key, dataset approval and available space; inspect the redacted download.log. No completion recorded.")
    archives = []
    for row in rows:
        matches = list(job.rglob(row["name"]))
        if len(matches) != 1 or (row["name"].lower().endswith(".zip") and not zipfile.is_zipfile(matches[0])):
            raise RuntimeError("Missing or incomplete archive after download; no completion recorded.")
        archives.append(str(matches[0]))
    record = {"dataset_key": str(dataset_key), "files": rows, "archives": archives,
              "source": GUIDE_URL, "status": "downloaded_archives", "review_status": "unreviewed"}
    (job / "download_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record

#!/usr/bin/env python3
"""qwen image studio — a local web UI for the ``qwen-image-2-1`` CLI, run under helmstudio.

Standard library only, plus helmstudio's runtime SDK (as ltx studio's web/server.py).
Each render spawns ``python -m qwen_image_2_1.generate`` with an argv built from the
form; one job runs at a time (one GPU) and the rest wait in a queue. The CLI's
``@stage`` / ``@step`` / ``@enhanced`` lines drive the page's progress over SSE.

qwen image studio keeps nothing of its own: sessions with their settings and
reference images, the page's paths (kv), references and takes (assets), takes with
their settings (gallery), and renders with their logs (jobs) all go to helmstudio.

Usage:
    helm dev -f helmstudio.yaml -venv .venv -link qwen_image=... (see web/run.sh)
"""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
import json
import mimetypes
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO, ClassVar
from urllib.parse import parse_qs, unquote, urlparse

from helm_runtime_sdk import HelmError, from_env
from helm_runtime_sdk.proxy import PREFIX, Proxy

WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEB_DIR.parent
STATIC_DIR = WEB_DIR / "static"

#: The CLI's --ratio choices (qwen_image_2_1.generate.ASPECT_RATIOS).
RATIOS = {"1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"}
MAX_INPUTS = 10
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

ASSETS = f"{PREFIX}api/v1/assets/"
JOB_STATES = {"done": "succeeded", "failed": "failed", "cancelled": "cancelled"}
PATHS_KEY = ("ui", "paths")
STAGES = ("enhance", "load", "denoise", "decode", "save")
#: Overall percent at the start and end of each stage, for helmstudio's job bar.
BANDS = {"enhance": (0, 20), "load": (20, 30), "denoise": (30, 92), "decode": (92, 98), "save": (98, 100)}

#: A reference image named in the prompt: ``@`` then its name, as the page inserts it.
MENTION_RE = re.compile(r"@([A-Za-z0-9._-]*[A-Za-z0-9_-])")
STAGE_RE = re.compile(r"^@stage (\w+)$")
STEP_RE = re.compile(r"^@step (\d+)/(\d+)$")
ENHANCED_RE = re.compile(r"^@enhanced (\{.*\})$")
SAVED_RE = re.compile(r"^Saved (.+) \((\d+)x(\d+), mode=(\w+)\)$")
LOG_COLOURS = {"cmd": "36", "done": "36", "failed": "31", "cancelled": "31"}


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def safe_name(text: str, default: str = "untitled") -> str:
    return SAFE_NAME.sub("-", text.strip()).strip(".-")[:80] or default


def asset_url(asset_id: str) -> str:
    return f"{ASSETS}{asset_id}"


def thumb_url(asset_id: str) -> str:
    return f"{ASSETS}{asset_id}/thumb?w=320"


def pages(fetch: Callable[..., dict[str, Any]]) -> Iterator[dict[str, Any]]:
    cursor = None
    while True:
        page = fetch(limit=200, cursor=cursor)
        yield from page.get("items") or []
        cursor = page.get("next_cursor")
        if not cursor:
            return


def image_size(path: Path) -> tuple[int, int] | None:
    from PIL import Image

    try:
        with Image.open(path) as img:
            return img.size
    except OSError:
        return None


def _quote(arg: str) -> str:
    return arg if re.fullmatch(r"[A-Za-z0-9_./:=@+,-]+", arg) else "'" + arg.replace("'", "'\\''") + "'"


# ---------------------------------------------------------------------------
# request guard (as ltx studio's): no auth, so refuse DNS rebinding and cross-site writes
# ---------------------------------------------------------------------------


def host_only(hostport: str) -> str:
    if hostport.startswith("["):
        return hostport[1:].split("]", 1)[0]
    return hostport.rsplit(":", 1)[0] if hostport.count(":") == 1 else hostport


def host_allowed(hostport: str, allowed: set[str]) -> bool:
    host = host_only(hostport).lower()
    if not host:
        return False
    if host == "localhost" or host in allowed:
        return True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def request_guard(method: str, path: str, headers: Any, allowed: set[str]) -> tuple[int, str] | None:
    host = headers.get("Host", "")
    if not host_allowed(host, allowed):
        return 403, f"unrecognized Host header; start with --allow-host {host_only(host)} to allow it"
    if method in {"GET", "HEAD"}:
        return None
    origin = headers.get("Origin")
    if origin and (origin == "null" or urlparse(origin).netloc.lower() != host.lower()):
        return 403, "cross-origin request refused"
    if not origin and headers.get("Sec-Fetch-Site") == "cross-site":
        return 403, "cross-site request refused"
    route = urlparse(path).path
    if route == "/api/upload":
        return None if headers.get("X-Filename") else (400, "X-Filename header is required")
    if route.startswith(PREFIX):
        return None
    if (headers.get("Content-Type") or "").split(";", 1)[0].strip().lower() != "application/json":
        return 415, "Content-Type must be application/json"
    return None


# ---------------------------------------------------------------------------
# helmstudio job: a render's state, progress and log
# ---------------------------------------------------------------------------


class HelmJob:
    """A render reported to helmstudio as a task job. Reporting never fails a render."""

    def __init__(self, client: Any, session_id: str, local_id: str) -> None:
        self.client, self.local_id = client, local_id
        self.id = client.jobs.create({"state": "queued", "subject_kind": "session", "subject_id": session_id})["id"]
        self.cancelling = False
        self._lock = threading.Lock()
        self._unsent: list[str] = []
        self._ended = threading.Event()
        self._progressed = 0.0

    def start(self) -> None:
        self._call(self.client.jobs.update, self.id, {"state": "running"})
        threading.Thread(target=self._send_log, daemon=True).start()

    def progress(self, percent: int) -> None:
        if time.monotonic() - self._progressed >= 1.0:
            self._progressed = time.monotonic()
            self._call(self.client.jobs.update, self.id, {"progress_num": percent, "progress_den": 100})

    def log(self, line: str, rewrites: bool = False) -> None:
        text = f"{line}\r" if rewrites else line
        with self._lock:
            if self._unsent and self._unsent[-1].endswith("\r"):
                self._unsent[-1] = text  # a progress line not sent yet gives way to the next
            else:
                self._unsent.append(text)

    def _send_log(self) -> None:
        while not self._ended.wait(0.5):
            self.flush()

    def flush(self) -> None:
        with self._lock:
            unsent, self._unsent = self._unsent, []
        for start in range(0, len(unsent), 1000):
            self._call(self.client.jobs.append_log, self.id, {"lines": [s[:16384] for s in unsent[start : start + 1000]]})

    def finish(self, status: str, error: str | None = None) -> None:
        self._ended.set()
        self.flush()
        body: dict[str, Any] = {"state": JOB_STATES.get(status, "failed")}
        if body["state"] == "succeeded":
            body.update(progress_num=100, progress_den=100)
        elif body["state"] == "failed":
            body["last_error"] = {"code": "render_failed", "message": (error or "the render failed")[:4000]}
        self._call(self.client.jobs.update, self.id, body)

    @staticmethod
    def _call(method: Callable[..., Any], *args: Any) -> None:
        try:
            method(*args)
        except Exception as exc:
            print(f"[helmstudio] {exc}", flush=True)


# ---------------------------------------------------------------------------
# state: everything helmstudio keeps for the studio
# ---------------------------------------------------------------------------


class State:
    def __init__(self, paths: dict[str, str], client: Any, proxy: Any) -> None:
        self.client, self.proxy = client, proxy
        where = client.me.get()["paths"]
        self.stage = Path(where["stage"])  # scratch, cleared when helmstudio stops the studio
        self.lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._renders: dict[str, HelmJob] = {}
        self.running: dict[str, Any] | None = None
        # The manifest's weights are the defaults; a path chosen in the page wins.
        self.paths = {**paths, **{k: v for k, v in self._kv(PATHS_KEY).items() if k in paths and v}}
        latest = next(iter(client.sessions.list(limit=1).get("items") or []), None)
        self.active = latest["name"] if latest else "session-1"

    def _kv(self, key: tuple[str, str]) -> dict[str, Any]:
        try:
            return self.client.kv.get(*key)["doc"]
        except HelmError as exc:
            if exc.status == 404:
                return {}
            raise

    # paths ----------------------------------------------------------------
    def paths_info(self) -> dict[str, Any]:
        return {k: {"path": v, "ok": bool(v) and Path(v).expanduser().is_dir()} for k, v in self.paths.items()}

    def set_paths(self, changes: dict[str, Any]) -> dict[str, Any]:
        for key, value in changes.items():
            if key not in self.paths:
                raise ValueError(f"unknown path: {key}")
            value = str(value).strip()
            if value and not Path(value).expanduser().is_dir():
                raise ValueError(f"{key}: no directory at {value}")
            self.paths[key] = value
        self.client.kv.put(*PATHS_KEY, self.paths)
        return self.paths_info()

    # sessions -------------------------------------------------------------
    def sessions(self) -> list[str]:
        with self.lock:
            self._sessions = {s["name"]: s for s in pages(self.client.sessions.list)}
            return sorted(self._sessions)

    def session(self, name: str, *, create: bool = False) -> dict[str, Any] | None:
        with self.lock:
            if name not in self._sessions:
                self.sessions()
            if name not in self._sessions and create:
                self._sessions[name] = self.client.sessions.create({"name": name, "state": {}})
            return self._sessions.get(name)

    def session_name(self, requested: Any) -> str:
        return str(requested or self.active).strip() or "session-1"

    def activate(self, name: str) -> dict[str, Any]:
        session = self.session(name, create=True)
        self.client.sessions.activate(session["id"])
        self.active = session["name"]
        return {"name": self.active, "settings": self._state(self.active).get("settings") or {}}

    def duplicate(self, name: str, new_name: str) -> dict[str, Any]:
        if not new_name.strip():
            raise ValueError("enter a name")
        try:
            copy = self.client.sessions.duplicate(self._existing(name)["id"], {"name": new_name.strip()})
        except HelmError as exc:
            if exc.status == 409:
                raise ValueError("a session with that name already exists") from None
            raise
        with self.lock:
            self._sessions[copy["name"]] = copy
        return self.activate(copy["name"])

    def delete_session(self, name: str) -> dict[str, Any]:
        """Its takes stay in helmstudio's gallery."""
        self.client.sessions.delete(self._existing(name)["id"])
        with self.lock:
            self._sessions.pop(name, None)
        remaining = self.sessions()
        return self.activate(remaining[0] if remaining else "session-1")

    def _existing(self, name: str) -> dict[str, Any]:
        session = self.session(name)
        if session is None:
            raise ValueError(f"no session named {name!r}")
        return session

    def _state(self, name: str) -> dict[str, Any]:
        return (self.session(name) or {}).get("state") or {}

    def _patch_state(self, name: str, change: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        """Merge ``change(state)`` into a session's state, retrying when another writer got there first."""
        with self.lock:
            for attempt in range(3):
                session = self.session(name, create=True)
                patch = change(session.get("state") or {})
                if not patch:
                    return
                try:
                    updated = self.client.sessions.update(session["id"], {"state": patch}, if_match=session["etag"])
                except HelmError as exc:
                    if exc.status not in (404, 409) or attempt == 2:
                        raise
                    self._sessions.pop(name, None)
                    continue
                self._sessions[updated["name"]] = updated
                return

    def save_settings(self, name: str, settings: dict[str, Any]) -> None:
        # A merge patch: null removes, so a setting left empty is removed.
        def change(state: dict[str, Any]) -> dict[str, Any]:
            old = state.get("settings") or {}
            patch: dict[str, Any] = {k: None for k in old if settings.get(k) is None}
            patch.update({k: v for k, v in settings.items() if old.get(k) != v and v is not None})
            return {"settings": patch} if patch else {}

        self._patch_state(name, change)

    # references: the session's input images, in the order the CLI gets them
    def inputs(self, name: str) -> list[dict[str, Any]]:
        entries = (self._state(name).get("inputs") or {}).items()
        ordered = sorted(entries, key=lambda pair: pair[1].get("added") or "")
        return [{"name": key, **entry, "url": asset_url(entry["asset_id"]), "thumb": thumb_url(entry["asset_id"])}
                for key, entry in ordered]  # fmt: skip

    def _add_input(self, session: str, name: str, entry: dict[str, Any]) -> str:
        chosen = name

        def change(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal chosen
            taken = state.get("inputs") or {}
            if len(taken) >= MAX_INPUTS:
                raise ValueError(f"Qwen-Image-2.1 takes at most {MAX_INPUTS} reference images")
            stem, suffix = os.path.splitext(name)
            chosen = name
            while chosen in taken:
                chosen = f"{stem}-{uuid.uuid4().hex[:4]}{suffix}"
            return {"inputs": {chosen: entry}}

        self._patch_state(session, change)
        return chosen

    def upload(self, session: str, original: str, body: BinaryIO, length: int) -> dict[str, Any]:
        stem, suffix = os.path.splitext(original)
        if suffix.lower() not in IMAGE_TYPES:
            raise ValueError("references must be PNG, JPEG or WebP")
        path = self.staged("uploads", f"{safe_name(stem, 'reference')}{suffix.lower()}")
        with open(path, "wb") as f:
            while length > 0:
                chunk = body.read(min(1 << 20, length))
                if not chunk:
                    break
                f.write(chunk)
                length -= len(chunk)
        size = image_size(path)
        if size is None:
            raise ValueError(f"{original} is not an image")
        asset = self.adopt(path, size, pinned=True)
        with contextlib.suppress(OSError):
            path.parent.rmdir()
        entry = {"asset_id": asset["id"], "original_name": original, "width": size[0], "height": size[1],
                 "added": now_iso()}  # fmt: skip
        return {"name": self._add_input(session, path.name, entry)}

    def delete_input(self, session: str, name: str) -> None:
        """Its asset stays pinned: takes made from it name it as an input."""
        self._patch_state(session, lambda s: {"inputs": {name: None}} if name in (s.get("inputs") or {}) else {})

    def take_as_input(self, session: str, take_id: str) -> dict[str, Any]:
        item = self.client.gallery.get(take_id)
        params = item.get("params") or {}
        path = self.staged("pin", params.get("name") or f"{take_id}.png")
        os.link(self.materialise(item["asset_id"], path.name), path)
        self.adopt(path, None, pinned=True)  # adopting the same bytes again pins the asset
        entry = {"asset_id": item["asset_id"], "original_name": path.name, "width": params.get("width"),
                 "height": params.get("height"), "added": now_iso()}  # fmt: skip
        return {"name": self._add_input(session, path.name, entry)}

    # assets ---------------------------------------------------------------
    def staged(self, kind: str, name: str) -> Path:
        path = self.stage / kind / uuid.uuid4().hex / Path(name).name
        path.parent.mkdir(parents=True)
        return path

    def adopt(self, path: Path, size: tuple[int, int] | None, *, pinned: bool = False) -> dict[str, Any]:
        hints = {"width": size[0], "height": size[1]} if size else {}
        return self.client.assets.adopt({"path": str(path), "kind": "image", "pinned": pinned, **hints})

    def materialise(self, asset_id: str, name: str) -> Path:
        """An asset as a file the CLI can read, fetched into the stage directory once per launch."""
        path = self.stage / "assets" / asset_id / Path(name).name
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(f"{path.name}.part")
            partial.write_bytes(self.client.assets.read(asset_id).read())
            partial.replace(path)
        return path

    # takes: the session's image items in helmstudio's gallery
    def takes(self, session: str) -> list[dict[str, Any]]:
        found = self.session(session)
        if found is None:
            return []
        items = pages(lambda **page: self.client.gallery.query(session_id=found["id"], kind="image", **page))
        return [{"id": item["id"], "url": asset_url(item["asset_id"]), "thumb": thumb_url(item["asset_id"]),
                 "starred": bool(item.get("starred")), "params": item.get("params") or {}}
                for item in items]  # fmt: skip

    def delete_take(self, take_id: str) -> None:
        self.client.gallery.delete(take_id)

    def take_finished(self, job: dict[str, Any], output: Path) -> str:
        result = job["result"]
        asset = self.adopt(output, (result["width"], result["height"]))
        settings = job["settings"]
        params = {
            "name": output.name, "prompt": settings["prompt"], "enhanced_prompt": result.get("enhanced_prompt"),
            "pe_ratio": result.get("pe_ratio"), "seed": settings["seed"], "steps": settings["steps"],
            "width": result["width"], "height": result["height"], "mode": result["mode"],
            "edit": bool(job["inputs"]), "created": job["created"], "elapsed": job["elapsed"], "settings": settings,
        }  # fmt: skip
        item = self.client.gallery.add({
            "kind": "image", "asset_id": asset["id"], "title": output.stem,
            "session_id": self.session(job["session"], create=True)["id"],
            "params": {k: v for k, v in params.items() if v not in (None, "")},
            "inputs": [{"asset_id": i["asset_id"], "role": "input"} for i in job["inputs"]],
        })  # fmt: skip
        return f"[helmstudio] kept {output.name} as gallery item {item['id']}"

    # renders as helmstudio jobs ---------------------------------------------
    def job_queued(self, job: dict[str, Any]) -> None:
        job["helm_job"] = None
        try:
            report = HelmJob(self.client, self.session(job["session"], create=True)["id"], job["id"])
        except Exception as exc:
            print(f"[helmstudio] could not report render {job['id']}: {exc}", flush=True)
            return
        self._renders[job["id"]] = report
        job["helm_job"] = report.id

    def job_started(self, job: dict[str, Any]) -> None:
        self.running = job
        if report := self._renders.get(job["id"]):
            report.start()

    def job_progress(self, job: dict[str, Any]) -> None:
        if report := self._renders.get(job["id"]):
            report.progress(percent(job["progress"]))

    def job_log(self, job: dict[str, Any], line: str, rewrites: bool = False, kind: str | None = None) -> None:
        if report := self._renders.get(job["id"]):
            colour = LOG_COLOURS.get(kind or "")
            report.log(f"\x1b[{colour}m{line}\x1b[0m" if colour else line, rewrites)

    def job_finished(self, job: dict[str, Any]) -> None:
        if self.running is job:
            self.running = None
        if report := self._renders.pop(job["id"], None):
            report.finish(job["status"], job.get("error"))

    def follow_cancellations(self, cancel: Callable[[str], Any]) -> None:
        """Cancel a render when helmstudio's launcher asks for it."""

        def follow() -> None:
            last = None
            while True:
                with contextlib.suppress(Exception):  # the stream ended: helmstudio restarted or is stopping us
                    for event in self.client.events.subscribe(last_event_id=last):
                        last = event.id or last
                        if event.name != "job":
                            continue
                        reported = (event.json() or {}).get("job") or {}
                        report = next((r for r in list(self._renders.values()) if r.id == reported.get("id")), None)
                        if report and reported.get("cancel_requested_at") and not report.cancelling:
                            report.cancelling = True
                            cancel(report.local_id)
                time.sleep(2)

        threading.Thread(target=follow, daemon=True).start()


def percent(progress: dict[str, Any]) -> int:
    low, high = BANDS.get(progress.get("stage") or "", (0, 0))
    if progress.get("stage") == "denoise" and progress.get("total"):
        return round(low + (high - low) * progress["step"] / progress["total"])
    return low


# ---------------------------------------------------------------------------
# runner: one render at a time
# ---------------------------------------------------------------------------


def as_int(value: Any, name: str, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number") from None
    if not low <= number <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return number


def resolve_mentions(prompt: str, names: list[str]) -> str:
    """``@name`` of a reference image → ``<imageN>``, its place in the list, as Qwen-Image-2.1 tags images.

    An ``@`` that names no reference (an email, a handle) stays as written.
    """
    index = {name: i for i, name in enumerate(names, 1)}
    return MENTION_RE.sub(lambda m: f"<image{index[m[1]]}>" if m[1] in index else m[0], prompt)


def clean_settings(req: dict[str, Any]) -> dict[str, Any]:
    """The form's settings, validated into what the CLI accepts."""
    prompt = str(req.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("write a prompt")
    ratio = req.get("ratio") or None
    if ratio is not None and ratio not in RATIOS:
        raise ValueError(f"unknown aspect ratio {ratio}")
    settings: dict[str, Any] = {"prompt": prompt, "ratio": ratio}
    for key in ("width", "height"):
        if req.get(key) not in (None, ""):
            value = as_int(req[key], key, 256, 4096)
            if value % 32:
                raise ValueError(f"{key} must be a multiple of 32")
            settings[key] = value
    settings["steps"] = as_int(req.get("steps", 40), "steps", 1, 200)
    settings["seed"] = as_int(req.get("seed", 42), "seed", 0, 2**63 - 1)
    for flag in ("enhance", "think", "transparent"):
        settings[flag] = bool(req.get(flag))
    settings["think"] = settings["think"] and settings["enhance"]
    return settings


class Runner:
    def __init__(self, state: State) -> None:
        self.state = state
        self.jobs: dict[str, dict[str, Any]] = {}
        self.pending: list[str] = []
        self.current: str | None = None
        self.proc: subprocess.Popen | None = None
        self.cond = threading.Condition()
        self.subscribers: list[queue.Queue] = []
        self.sub_lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()
        state.follow_cancellations(self.cancel)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self.sub_lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.sub_lock, contextlib.suppress(ValueError):
            self.subscribers.remove(q)

    def emit(self, kind: str, payload: Any) -> None:
        message = json.dumps({"type": kind, "data": payload})
        with self.sub_lock:
            for q in self.subscribers:
                with contextlib.suppress(queue.Full):
                    q.put_nowait(message)

    def log(self, job: dict[str, Any], line: str, rewrites: bool = False, kind: str | None = None) -> None:
        self.state.job_log(job, line, rewrites, kind)
        self.emit("log", {"session": job["session"], "line": line, "rewrites": rewrites, "kind": kind})

    def summary(self, job: dict[str, Any]) -> dict[str, Any]:
        keys = ("id", "session", "status", "created", "elapsed", "error", "helm_job", "argv_display", "result")
        return {**{k: job.get(k) for k in keys}, "prompt": job["settings"]["prompt"],
                "ratio": job["settings"]["ratio"], "seed": job["settings"]["seed"],
                "progress": dict(job["progress"])}  # fmt: skip

    def queue_state(self) -> list[dict[str, Any]]:
        with self.cond:
            ids = ([self.current] if self.current else []) + self.pending
            return [self.summary(self.jobs[i]) for i in ids]

    def submit(self, req: dict[str, Any]) -> dict[str, Any]:
        state = self.state
        session = state.session_name(req.get("session"))
        settings = clean_settings(req)
        inputs = state.inputs(session)
        if not (state.paths["model"] and Path(state.paths["model"]).expanduser().is_dir()):
            raise ValueError("the image model path is not set — choose it under Paths")
        pe_key = "pe_i2i" if inputs else "pe_t2i"
        if settings["enhance"] and not Path(state.paths[pe_key] or "/nonexistent").expanduser().is_dir():
            raise ValueError(f"the prompt enhancer path ({pe_key}) is not set — choose it under Paths")
        job_id = uuid.uuid4().hex[:10]
        output = state.stage / "takes" / f"take-{datetime.now():%m%d-%H%M%S}-{job_id[:4]}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        prompt = resolve_mentions(settings["prompt"], [i["name"] for i in inputs])
        argv = [prompt, "-m", os.path.expanduser(state.paths["model"]), "--output", str(output),
                "--steps", str(settings["steps"]), "--seed", str(settings["seed"])]  # fmt: skip
        if settings["ratio"]:
            argv += ["--ratio", settings["ratio"]]
        for key in ("width", "height"):
            if key in settings:
                argv += [f"--{key}", str(settings[key])]
        for flag in ("enhance", "think", "transparent"):
            if settings[flag]:
                argv.append(f"--{flag}")
        if settings["enhance"]:
            argv += ["--pe-model", os.path.expanduser(state.paths[pe_key])]
        job = {
            "id": job_id, "session": session, "status": "queued", "created": now_iso(),
            "settings": settings, "inputs": inputs, "argv": argv, "output": str(output),
            "argv_display": "qwen-image-2-1 " + " ".join(_quote(a) for a in argv),
            "progress": {"stage": None, "step": 0, "total": 0}, "result": {},
        }  # fmt: skip
        state.job_queued(job)
        with self.cond:
            self.jobs[job_id] = job
            self.pending.append(job_id)
            self.cond.notify()
        self.emit("queue", self.queue_state())
        return self.summary(job)

    def cancel(self, job_id: str) -> bool:
        with self.cond:
            job = self.jobs.get(job_id)
            if job_id in self.pending:
                self.pending.remove(job_id)
                job["status"] = "cancelled"
                self.state.job_finished(job)
                self.emit("queue", [self.summary(self.jobs[i]) for i in ([self.current] if self.current else []) + self.pending])
                return True
            if job_id == self.current and self.proc and self.proc.poll() is None:
                job["status"] = "cancelling"
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(self.proc.pid, signal.SIGTERM)
                return True
        return False

    def _loop(self) -> None:
        while True:
            with self.cond:
                while not self.pending:
                    self.cond.wait()
                self.current = self.pending.pop(0)
                job = self.jobs[self.current]
            try:
                self._run(job)
            except Exception as exc:
                job["status"], job["error"] = "failed", str(exc)
                self.log(job, f"[studio] {exc}", kind="failed")
            finally:
                with self.cond:
                    self.current, self.proc = None, None
                self.state.job_finished(job)
                self.emit("job", self.summary(job))
                self.emit("queue", self.queue_state())

    def _run(self, job: dict[str, Any]) -> None:
        job["status"] = "running"
        self.state.job_started(job)
        started = time.monotonic()
        self.emit("queue", self.queue_state())
        self.log(job, f"$ {job['argv_display']}", kind="cmd")
        # Every reference, materialised for the CLI, in the session's order.
        paths = [str(self.state.materialise(i["asset_id"], i["name"])) for i in job["inputs"]]
        argv = job["argv"] + (["--input", ",".join(paths)] if paths else [])
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TQDM_MININTERVAL": "0.5"}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "qwen_image_2_1.generate", *argv], cwd=REPO_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        )  # fmt: skip
        tail: list[str] = []
        self._pump(job, self.proc.stdout, tail)  # type: ignore[arg-type]
        returncode = self.proc.wait()
        job["elapsed"] = round(time.monotonic() - started, 1)
        output = Path(job["output"])
        if job["status"] == "cancelling":
            job["status"] = "cancelled"
        elif returncode == 0 and output.is_file() and job["result"].get("width"):
            job["status"] = "done"
            try:
                self.log(job, self.state.take_finished(job, output))
            except Exception as exc:
                job["status"], job["error"] = "failed", f"{output.name} could not be kept: {exc}"
            self.emit("takes", {"session": job["session"]})
        else:
            job["status"] = "failed"
            job["error"] = next((line for line in reversed(tail) if line.strip()), f"exit code {returncode}")
        output.unlink(missing_ok=True)  # adopted by hardlink, or not worth keeping
        self.log(job, f"[studio] {job['status']} in {job['elapsed']}s", kind=job["status"])

    def _pump(self, job: dict[str, Any], stream: BinaryIO, tail: list[str]) -> None:
        buf = b""
        while chunk := stream.read1(4096):  # type: ignore[attr-defined]
            buf += chunk
            while (idx := min((i for i in (buf.find(b"\n"), buf.find(b"\r")) if i != -1), default=-1)) != -1:
                rewrites = buf[idx : idx + 1] == b"\r"
                line, buf = buf[:idx].decode("utf-8", "replace"), buf[idx + 1 :]
                if not line.strip():
                    continue
                if self._parse(job, line.strip()):
                    self.emit("progress", {"id": job["id"], "progress": job["progress"], "result": job["result"]})
                    self.state.job_progress(job)
                if not line.startswith("@"):
                    if not rewrites:
                        tail.append(line)
                        del tail[:-40]
                    self.log(job, line, rewrites)
        if buf.strip():
            self.log(job, buf.decode("utf-8", "replace"))

    @staticmethod
    def _parse(job: dict[str, Any], text: str) -> bool:
        progress, result = job["progress"], job["result"]
        if m := STAGE_RE.match(text):
            if m[1] in STAGES:
                progress.update(stage=m[1], step=0, total=0)
            return True
        if m := STEP_RE.match(text):
            progress.update(step=int(m[1]), total=int(m[2]))
            return True
        if m := ENHANCED_RE.match(text):
            with contextlib.suppress(json.JSONDecodeError):
                data = json.loads(m[1])
                result.update(enhanced_prompt=data.get("prompt"), pe_ratio=data.get("ratio"))
            return True
        if m := SAVED_RE.match(text):
            result.update(width=int(m[2]), height=int(m[3]), mode=m[4])
            return True
        return False


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "qwen-image-studio"
    state: State
    runner: Runner
    allowed_hosts: ClassVar[set[str]] = set()

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _error(self, message: str, code: int = 400) -> None:
        self._json({"error": message, "message": message}, code)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}") if length else {}

    def _guarded(self) -> bool:
        """True when the request was answered: refused, or served by helmstudio's /helm/ proxy."""
        refusal = request_guard(self.command, self.path, self.headers, self.allowed_hosts)
        if refusal:
            self._error(refusal[1], refusal[0])
            return True
        return urlparse(self.path).path.startswith(PREFIX) and self.state.proxy.handle_http(self)

    def _route(self, handler: Callable[[str, dict[str, str]], Any]) -> None:
        if self._guarded():
            return
        url = urlparse(self.path)
        try:
            result = handler(url.path, {k: v[0] for k, v in parse_qs(url.query).items()})
            if result is not None:
                self._json(result)
        except (ValueError, KeyError) as exc:
            self._error(str(exc))
        except HelmError as exc:
            self._error(f"helmstudio: {exc}", 502)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # never drop the connection without a response
            self._error(f"{type(exc).__name__}: {exc}", 500)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_PUT(self) -> None:
        if not self._guarded():
            self._error("not found", 404)

    do_PATCH = do_DELETE = do_PUT

    def do_GET(self) -> None:
        self._route(self._get)

    def do_POST(self) -> None:
        self._route(self._post)

    def _get(self, path: str, query: dict[str, str]) -> Any:
        state, session = self.state, self.state.session_name(query.get("session"))
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path[len("/static/") :])
        if path == "/api/events":
            return self._events()
        if path == "/api/config":
            return {"paths": state.paths_info(), "sessions": state.sessions(), "active": state.active,
                    "settings": state._state(state.active).get("settings") or {}}  # fmt: skip
        if path == "/api/inputs":
            return state.inputs(session)
        if path == "/api/takes":
            return state.takes(session)
        if path == "/api/queue":  # also helmstudio's busy probe
            return self.runner.queue_state()
        return self._error("not found", 404)

    def _post(self, path: str, query: dict[str, str]) -> Any:
        state = self.state
        if path == "/api/upload":
            original = Path(unquote(self.headers.get("X-Filename") or "upload")).name
            length = int(self.headers.get("Content-Length") or 0)
            result = state.upload(state.session_name(query.get("session")), original, self.rfile, length)
            self.runner.emit("inputs", {})
            return result
        body = self._body()
        session = state.session_name(body.get("session"))
        if path == "/api/render":
            return self.runner.submit({**body, "session": session})
        if path == "/api/cancel":
            return {"ok": self.runner.cancel(str(body.get("id")))}
        if path == "/api/paths":
            return state.set_paths(body.get("paths") or {})
        if path == "/api/session/activate":
            return state.activate(session)
        if path == "/api/session/save":
            state.save_settings(session, body.get("settings") or {})
            return {"ok": True}
        if path == "/api/session/duplicate":
            return state.duplicate(session, str(body.get("new_name", "")))
        if path == "/api/session/delete":
            return state.delete_session(session)
        if path == "/api/inputs/delete":
            state.delete_input(session, str(body.get("name")))
            return {"ok": True}
        if path == "/api/takes/delete":
            state.delete_take(str(body["id"]))
            return {"ok": True}
        if path == "/api/takes/use":
            return state.take_as_input(session, str(body["id"]))
        return self._error("not found", 404)

    def _static(self, rel: str) -> None:
        target = (STATIC_DIR / rel).resolve()
        if STATIC_DIR not in target.parents or not target.is_file():
            return self._error("not found", 404)
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        return self._send(200, target.read_bytes(), ctype)

    def _events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        q = self.runner.subscribe()
        try:
            self.wfile.write(f"data: {json.dumps({'type': 'queue', 'data': self.runner.queue_state()})}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    self.wfile.write(f"data: {q.get(timeout=15)}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except OSError:
            pass
        finally:
            self.runner.unsubscribe(q)


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            super().handle_error(request, client_address)


def main() -> None:
    env = os.environ.get
    parser = argparse.ArgumentParser(description="qwen image studio")
    parser.add_argument("--model", default=env("QWEN_IMAGE_21_PATH", ""), help="Qwen-Image-2.1 directory")
    parser.add_argument("--pe-t2i", default=env("QWEN_IMAGE_21_PE_T2I_PATH", ""), help="PE-T2I directory")
    parser.add_argument("--pe-i2i", default=env("QWEN_IMAGE_21_PE_I2I_PATH", ""), help="PE-I2I directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8730)
    parser.add_argument("--allow-host", default="", help="extra Host names to accept, comma-separated")
    args = parser.parse_args()

    if not env("HELM_API"):
        sys.exit("qwen image studio: start it from helmstudio, or with `bash web/run.sh` (helm dev)")
    state = State({"model": args.model, "pe_t2i": args.pe_t2i, "pe_i2i": args.pe_i2i}, from_env(), Proxy.from_env())
    Handler.state, Handler.runner = state, Runner(state)
    Handler.allowed_hosts = {h.strip().lower() for h in [args.host, *args.allow_host.split(",")] if h.strip()}
    print(f"qwen image studio on http://{args.host}:{args.port}", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        StudioServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

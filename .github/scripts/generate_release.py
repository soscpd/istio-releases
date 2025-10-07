#!/usr/bin/env python3
"""Generate release metadata for Envoy and Istio."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.error import HTTPError
from urllib.request import Request, urlopen

CONFIG: Dict[str, Dict[str, object]] = {
    "envoyproxy/envoy": {
        "product": "envoy",
        "strip_v_dir": True,
        "images": [
            {
                "id": "base-image",
                "ref": "envoyproxy/envoy:{tag}",
                "registry": "envoyproxy/envoy",
            },
            {
                "id": "base-image.distroless",
                "ref": "envoyproxy/envoy:distroless-{tag_no_v}",
                "registry": "envoyproxy/envoy",
            },
        ],
        "publish": {
            "docker-images": {
                "fips": ["cloudsmith://docker-images/fips"],
                "non-fips": [
                    "cloudsmith://docker-images/addon",
                    "cloudsmith://docker-images/non-fips",
                ],
            },
            "linux-distribution-packages": {
                "fips": ["cloudsmith://linux-distribution-packages/fips"],
                "non-fips": ["cloudsmith://linux-distribution-packages/non-fips"],
            },
        },
        "release_notes": "https://www.envoyproxy.io/docs/envoy/v{tag_no_v}/version_history/v{major_minor}/v{tag_no_v}",
    },
    "istio/istio": {
        "product": "istio",
        "strip_v_dir": False,
        "images": [
            {
                "id": "base-image",
                "ref": "gcr.io/istio-release/pilot:{tag_no_v}",
                "registry": "gcr.io/istio-release/pilot",
            },
            {
                "id": "base-image.distroless",
                "ref": "gcr.io/istio-release/pilot:{tag_no_v}-distroless",
                "registry": "gcr.io/istio-release/pilot",
            },
        ],
        "publish": {
            "docker-images": {
                "fips": ["cloudsmith://docker-images/fips"],
                "non-fips": [
                    "cloudsmith://docker-images/addon",
                    "cloudsmith://docker-images/non-fips",
                ],
            },
            "linux-distribution-packages": {
                "fips": ["cloudsmith://linux-distribution-packages/fips"],
                "non-fips": ["cloudsmith://linux-distribution-packages/non-fips"],
            },
        },
        "release_notes": "https://github.com/istio/istio/releases/tag/{tag}",
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def get_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise SystemExit(f"Missing required environment variable: {name}")
    return value.strip()


def gh_request(path: str, token: Optional[str]) -> Optional[dict]:
    url = f"https://api.github.com{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "istio-releases-action",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    try:
        with urlopen(req) as resp:
            if resp.status == 204:
                return None
            data = resp.read()
            if not data:
                return None
            return json.loads(data.decode())
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def resolve_tag(owner: str, repo: str, candidates: Iterable[str], token: Optional[str]) -> Tuple[str, str]:
    for candidate in candidates:
        if not candidate:
            continue
        path = f"/repos/{owner}/{repo}/git/ref/tags/{candidate}"
        ref = gh_request(path, token)
        if not ref:
            continue
        obj = ref.get("object", {})
        sha = obj.get("sha")
        typ = obj.get("type")
        if not sha:
            continue
        if typ == "tag":
            tag_obj = gh_request(f"/repos/{owner}/{repo}/git/tags/{sha}", token)
            if not tag_obj:
                raise RuntimeError(f"Unable to resolve annotated tag for {candidate}")
            sha = (tag_obj.get("object") or {}).get("sha")
            if not sha:
                raise RuntimeError(f"Annotated tag missing object SHA for {candidate}")
        return candidate, sha
    raise SystemExit("Unable to resolve provided tag against GitHub API")


def semver_tuple(tag: str) -> Tuple[int, int, int]:
    cleaned = tag.lstrip("v")
    parts = cleaned.split(".")
    nums = []
    for part in parts[:3]:
        match = re.match(r"(\d+)", part)
        nums.append(int(match.group(1)) if match else 0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])


def semver_ge(a: str, b: str) -> bool:
    return semver_tuple(a) >= semver_tuple(b)


def parse_start_from(raw: Optional[str]) -> Tuple[dict, Optional[str]]:
    mapping: Dict[str, str] = {}
    global_default: Optional[str] = None
    if raw:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "@" in item:
                repo, tag = item.split("@", 1)
                mapping[repo.strip()] = tag.strip()
            else:
                global_default = item
    return mapping, global_default


def ensure_baseline(repo: str, tag: str, start_from_raw: Optional[str]) -> None:
    mapping, global_default = parse_start_from(start_from_raw)
    baseline = mapping.get(repo, global_default)
    if not baseline:
        return
    if not semver_ge(tag, baseline):
        log(f"Tag {tag} is before baseline {baseline}; nothing to do.")
        output = Path(os.environ["GITHUB_OUTPUT"])
        with output.open("a", encoding="utf-8") as fh:
            fh.write("changes=false\n")
        raise SystemExit(0)


def next_release_dir(base_dir: Path, version_part: str) -> Path:
    candidate = base_dir / f"{version_part}+0"
    return candidate


def inspect_digest(image_ref: str) -> str:
    cmd = [
        "skopeo",
        "inspect",
        "--tls-verify=true",
        "--format",
        "{{.Digest}}",
        f"docker://{image_ref}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Failed to inspect {image_ref}: {stderr}")
    return result.stdout.strip()


def dump_yaml(data: dict, destination: Path) -> None:
    proc = subprocess.run(
        ["yq", "-P", "."],
        input=json.dumps(data),
        text=True,
        capture_output=True,
        check=True,
    )
    destination.write_text(proc.stdout, encoding="utf-8")


def diff_files(old: Path, new: Path) -> str:
    if not old.exists() or not new.exists():
        return ""
    result = subprocess.run(
        ["diff", "-ur", old.as_posix(), new.as_posix()],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def find_previous_release(base_dir: Path, version_tuple: Tuple[int, int, int]) -> Optional[Path]:
    if not base_dir.exists():
        return None
    candidates: list[Tuple[Tuple[int, int, int], int, Path]] = []
    major_minor = version_tuple[:2]
    for path in base_dir.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if "+" not in name:
            continue
        base, plus = name.rsplit("+", 1)
        try:
            iteration = int(plus)
        except ValueError:
            continue
        parts = base.split(".")
        try:
            numbers = tuple(int(p) for p in parts)
        except ValueError:
            continue
        if len(numbers) < 3:
            numbers = numbers + (0,) * (3 - len(numbers))
        if numbers[:2] != major_minor:
            continue
        if numbers >= version_tuple:
            continue
        candidates.append((numbers, iteration, path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][2]


def main() -> None:
    repo_and_tag = get_env("RELEASE")
    token = os.environ.get("GITHUB_TOKEN", "").strip() or None
    start_from = os.environ.get("START_FROM")

    if "@" not in repo_and_tag:
        raise SystemExit("RELEASE input must be in the form owner/repo@tag")

    repo_full, provided_tag = repo_and_tag.split("@", 1)
    repo_full = repo_full.strip()
    provided_tag = provided_tag.strip()
    if repo_full not in CONFIG:
        raise SystemExit(f"Unsupported repository: {repo_full}")

    cfg = CONFIG[repo_full]
    owner, repo = repo_full.split("/")

    if provided_tag.startswith("v"):
        candidates = [provided_tag, provided_tag[1:]]
    else:
        candidates = [provided_tag, f"v{provided_tag}"]
    candidates = [c for i, c in enumerate(candidates) if c and c not in candidates[:i]]

    resolved_tag, sha = resolve_tag(owner, repo, candidates, token)
    log(f"Resolved {repo_full}@{resolved_tag} -> {sha}")

    ensure_baseline(repo_full, resolved_tag, start_from)

    tag_no_v = resolved_tag[1:] if resolved_tag.startswith("v") else resolved_tag
    tag_with_v = resolved_tag if resolved_tag.startswith("v") else f"v{resolved_tag}"

    product = cfg["product"]  # type: ignore[index]
    strip_v_dir = bool(cfg.get("strip_v_dir"))
    dir_version = tag_no_v if strip_v_dir else resolved_tag

    releases_root = Path("releases") / str(product)
    releases_root.mkdir(parents=True, exist_ok=True)

    release_dir = next_release_dir(releases_root, dir_version)
    if release_dir.exists():
        log(f"Release directory {release_dir} already exists; skipping generation.")
        output = Path(os.environ["GITHUB_OUTPUT"])
        with output.open("a", encoding="utf-8") as fh:
            fh.write("changes=false\n")
        return

    release_dir.mkdir(parents=True)

    images_info = {}
    for image in cfg["images"]:  # type: ignore[index]
        image_id = image["id"]
        ref_template = image["ref"]
        registry = image["registry"]
        ref = (
            ref_template
            .replace("{tag}", resolved_tag)
            .replace("{tag_no_v}", tag_no_v)
            .replace("{tag_with_v}", tag_with_v)
        )
        log(f"Inspecting {ref}")
        digest = inspect_digest(ref)
        images_info[image_id] = {
            "registry": registry,
            "ref": ref,
            "digest": digest,
        }

    release_ref = f"{repo_full}@{sha}"
    build_payload = {
        "base-image": {
            "registry": images_info["base-image"]["registry"],
            "version": images_info["base-image"]["digest"],
            "distroless": {
                "registry": images_info["base-image.distroless"]["registry"],
                "version": images_info["base-image.distroless"]["digest"],
            },
        },
        "release": {"ref": release_ref},
        "release-fips": {"ref": release_ref},
    }
    dump_yaml(build_payload, release_dir / "build.yaml")

    major_minor = ".".join(tag_no_v.split(".")[:2]) or tag_no_v
    publish_base = json.loads(json.dumps(cfg["publish"]))  # deep copy
    publish_base.setdefault("release-notes", {})["announcement"] = (
        cfg["release_notes"].format(
            tag=resolved_tag,
            tag_no_v=tag_no_v,
            tag_with_v=tag_with_v,
            major_minor=major_minor,
        )
    )
    dump_yaml(publish_base, release_dir / "publish.yaml")

    version_tuple = semver_tuple(resolved_tag)
    previous_dir = find_previous_release(releases_root, version_tuple)

    issue_lines = [f"Update to the latest {major_minor} minor.", ""]
    if previous_dir:
        prev_build = previous_dir / "build.yaml"
        prev_publish = previous_dir / "publish.yaml"
        build_diff = diff_files(prev_build, release_dir / "build.yaml")
        publish_diff = diff_files(prev_publish, release_dir / "publish.yaml")
        if build_diff:
            issue_lines.append(build_diff)
            issue_lines.append("")
        if publish_diff:
            issue_lines.append(publish_diff)
    else:
        issue_lines.append("No previous release available for diff.")
    issue_text = "\n".join(issue_lines).rstrip() + "\n"
    (release_dir / "issue.md").write_text(issue_text, encoding="utf-8")

    output = Path(os.environ["GITHUB_OUTPUT"])
    branch_slug = dir_version.replace("+", "-plus-")
    with output.open("a", encoding="utf-8") as fh:
        fh.write("changes=true\n")
        fh.write(f"product={product}\n")
        fh.write(f"version_dir={dir_version}+0\n")
        fh.write(f"release_tag={resolved_tag}\n")
        fh.write(f"release_dir={release_dir.as_posix()}\n")
        fh.write(f"pr_branch=release/{product}-{branch_slug}\n")
        fh.write(f"commit_message=chore: add {product} {dir_version}+0 release metadata\n")
        fh.write(f"pr_title=Add {product} {dir_version}+0 release metadata\n")

    log(f"Generated release metadata in {release_dir}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            log(str(exc))
        raise

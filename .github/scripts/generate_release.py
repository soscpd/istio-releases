#!/usr/bin/env python3
"""Generate release metadata for Envoy and Istio."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
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
                "ref": "envoyproxy/envoy:distroless-{tag_with_v}",
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
def gh_request(path: str, token: Optional[str]) -> Optional[Any]:
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


def list_repo_tags(owner: str, repo: str, token: Optional[str]) -> List[Tuple[str, str]]:
    tags: List[Tuple[str, str]] = []
    page = 1
    while True:
        path = f"/repos/{owner}/{repo}/tags?per_page=100&page={page}"
        response = gh_request(path, token)
        if not response:
            break
        if not isinstance(response, list):
            raise RuntimeError(f"Unexpected response for tag listing: {response!r}")
        for item in response:
            name = item.get("name") if isinstance(item, dict) else None
            commit = item.get("commit") if isinstance(item, dict) else None
            sha = commit.get("sha") if isinstance(commit, dict) else None
            if name and sha:
                tags.append((name, sha))
        if len(response) < 100:
            break
        page += 1
    return tags


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


def next_release_dir(base_dir: Path, version_part: str) -> Path:
    candidate = base_dir / f"{version_part}+0"
    return candidate


def release_dir_exists(base_dir: Path, version_part: str) -> bool:
    if not base_dir.exists():
        return False
    prefix = f"{version_part}+"
    for path in base_dir.iterdir():
        if path.is_dir() and path.name.startswith(prefix):
            return True
    return False


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


def write_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    value = str(value)
    with Path(output_path).open("a", encoding="utf-8") as fh:
        if "\n" in value:
            fh.write(f"{name}<<EOF\n{value}\nEOF\n")
        else:
            fh.write(f"{name}={value}\n")


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "release"


def generate_release(
    repo_full: str,
    resolved_tag: str,
    sha: str,
    cfg: Dict[str, object],
) -> Optional[Dict[str, str]]:
    product = cfg["product"]  # type: ignore[index]
    strip_v_dir = bool(cfg.get("strip_v_dir"))
    tag_no_v = resolved_tag[1:] if resolved_tag.startswith("v") else resolved_tag
    tag_with_v = resolved_tag if resolved_tag.startswith("v") else f"v{resolved_tag}"
    dir_version = tag_no_v if strip_v_dir else resolved_tag

    releases_root = Path("releases") / str(product)
    releases_root.mkdir(parents=True, exist_ok=True)

    release_dir = next_release_dir(releases_root, dir_version)
    if release_dir.exists():
        log(f"Release directory {release_dir} already exists; skipping generation.")
        return None

    release_dir.mkdir(parents=True)

    images_info: Dict[str, Dict[str, str]] = {}
    for image in cfg["images"]:  # type: ignore[index]
        image_id = image["id"]  # type: ignore[index]
        ref_template = image["ref"]  # type: ignore[index]
        registry = image["registry"]  # type: ignore[index]
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
        cfg["release_notes"].format(  # type: ignore[index]
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

    log(f"Generated release metadata in {release_dir}")

    return {
        "product": str(product),
        "dir_version": release_dir.name,
        "release_dir": release_dir.as_posix(),
        "release_tag": resolved_tag,
    }


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN", "").strip() or None
    start_from = os.environ.get("START_FROM")
    release_env = os.environ.get("RELEASE", "").strip()

    if release_env:
        if "@" not in release_env:
            raise SystemExit("RELEASE input must be in the form owner/repo@tag")

        repo_full, provided_tag = release_env.split("@", 1)
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

        mapping, global_default = parse_start_from(start_from)
        baseline = mapping.get(repo_full, global_default)
        if baseline and not semver_ge(resolved_tag, baseline):
            log(f"Tag {resolved_tag} is before baseline {baseline}; nothing to do.")
            write_output("changes", "false")
            return

        entry = generate_release(repo_full, resolved_tag, sha, cfg)
        if not entry:
            write_output("changes", "false")
            return

        version_slug = entry["dir_version"].replace("+", "-plus-")
        branch_slug = slugify(f"{entry['product']}-{version_slug}")
        write_output("changes", "true")
        write_output("product", entry["product"])
        write_output("version_dir", entry["dir_version"])
        write_output("release_tag", entry["release_tag"])
        write_output("release_dir", entry["release_dir"])
        write_output("paths", entry["release_dir"])
        write_output("summary", f"Generated {entry['product']} release at {entry['release_dir']}")
        write_output("pr_branch", f"release/{branch_slug}")
        write_output(
            "commit_message",
            f"chore: add {entry['product']} {entry['dir_version']} release metadata",
        )
        write_output(
            "pr_title",
            f"Add {entry['product']} {entry['dir_version']} release metadata",
        )
        body = (
            "Automated release metadata for `"
            f"{entry['release_tag']}`.\n\nGenerated directories:\n- `"
            f"{entry['release_dir']}`"
        )
        write_output("pr_body", body)
        return

    mapping, global_default = parse_start_from(start_from)
    generated: List[Dict[str, str]] = []

    for repo_full, cfg in CONFIG.items():
        owner, repo = repo_full.split("/")
        baseline = mapping.get(repo_full, global_default)
        tags = list_repo_tags(owner, repo, token)
        if not tags:
            log(f"No tags found for {repo_full}")
            continue

        tags.sort(key=lambda item: semver_tuple(item[0]))

        product = cfg["product"]  # type: ignore[index]
        strip_v_dir = bool(cfg.get("strip_v_dir"))
        releases_root = Path("releases") / str(product)

        for tag_name, _sha in tags:
            if not re.match(r"^v?\d+\.\d+\.\d+", tag_name):
                continue
            if baseline and not semver_ge(tag_name, baseline):
                continue
            if tag_name.startswith("v"):
                candidates = [tag_name, tag_name[1:]]
            else:
                candidates = [tag_name, f"v{tag_name}"]
            candidates = [c for i, c in enumerate(candidates) if c and c not in candidates[:i]]

            resolved_tag, sha = resolve_tag(owner, repo, candidates, token)
            log(f"Resolved {repo_full}@{resolved_tag} -> {sha}")
            tag_no_v = resolved_tag[1:] if resolved_tag.startswith("v") else resolved_tag
            dir_version = tag_no_v if strip_v_dir else resolved_tag
            if release_dir_exists(releases_root, dir_version):
                continue
            entry = generate_release(repo_full, resolved_tag, sha, cfg)
            if entry:
                generated.append(entry)

    if not generated:
        write_output("changes", "false")
        return

    first = generated[0]
    count = len(generated)
    version_slug = first["dir_version"].replace("+", "-plus-")
    if count == 1:
        branch_slug = slugify(f"{first['product']}-{version_slug}")
        commit_message = f"chore: add {first['product']} {first['dir_version']} release metadata"
        pr_title = f"Add {first['product']} {first['dir_version']} release metadata"
        body = (
            "Automated release metadata for `"
            f"{first['release_tag']}`.\n\nGenerated directories:\n- `"
            f"{first['release_dir']}`"
        )
        summary = f"Generated {first['product']} release at {first['release_dir']}"
    else:
        branch_slug = slugify(
            f"batch-{first['product']}-{version_slug}-and-{count - 1}-more"
        )
        commit_message = f"chore: add release metadata for {count} updates"
        pr_title = f"Add release metadata for {count} updates"
        lines = [
            f"- **{entry['product']}** `{entry['release_tag']}` -> `{entry['release_dir']}`"
            for entry in generated
        ]
        body = (
            "Automated release metadata for the following updates:\n"
            + "\n".join(lines)
        )
        summary_lines = [
            f"- {entry['product']} {entry['dir_version']}" for entry in generated
        ]
        summary = f"Generated {count} release updates:\n" + "\n".join(summary_lines)

    paths = "\n".join(entry["release_dir"] for entry in generated)

    write_output("changes", "true")
    write_output("product", first["product"])
    write_output("version_dir", first["dir_version"])
    write_output("release_tag", first["release_tag"])
    write_output("release_dir", first["release_dir"])
    write_output("paths", paths)
    write_output("summary", summary)
    write_output("pr_branch", f"release/{branch_slug}")
    write_output("commit_message", commit_message)
    write_output("pr_title", pr_title)
    write_output("pr_body", body)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            log(str(exc))
        raise

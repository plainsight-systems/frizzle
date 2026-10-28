#!/usr/bin/env python3
"""Fetches native Dawn, prebuilt by its own release CI, as pinned in
cmake/dawn.json.

The native build links Dawn so that GPU code, which the browser runs through
emdawnwebgpu, is also tested natively: the same webgpu.h, from the same Dawn
release. This downloads this platform's prebuilt archive, verifies it against
its pinned SHA-256, and unpacks it into .cache/dawn/<release>/<platform>/, the
directory CMakeLists.txt reads the same pin file to find. An archive that does
not match its pin is not unpacked, and the run fails. One already unpacked is
not fetched again.

Standard library only, so CI runs it without installing anything.

    python3 scripts/fetch_dawn.py
"""
import hashlib
import io
import json
import pathlib
import platform
import shutil
import sys
import tarfile
import urllib.request

PINS = pathlib.Path("cmake/dawn.json")
DESTINATION = pathlib.Path(".cache/dawn")


def platform_key():
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    return f"{platform.system()}-{machine}"


def main():
    pins = json.loads(PINS.read_text())
    key = platform_key()
    if key not in pins["platforms"]:
        sys.exit(f"no Dawn build is pinned for {key}; cmake/dawn.json lists "
                 f"{', '.join(pins['platforms'])}")
    pin = pins["platforms"][key]
    target = DESTINATION / pins["release"] / key
    marker = target / ".sha256"
    if marker.is_file() and marker.read_text() == pin["sha256"]:
        return 0

    url = f"https://github.com/google/dawn/releases/download/{pins['release']}/{pin['asset']}"
    with urllib.request.urlopen(url) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != pin["sha256"]:
        sys.exit(f"{pin['asset']}: SHA-256 {digest} does not match the pinned "
                 f"{pin['sha256']}; not unpacked")

    # Unpacked beside the target, then moved into place, so a half-unpacked
    # tree is never mistaken for Dawn. The archive's one top-level directory
    # is dropped: the target is the same path on every platform.
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        top = pin["asset"].removesuffix(".tar.gz") + "/"
        for member in archive.getmembers():
            if member.name == top.rstrip("/"):
                continue
            if not member.name.startswith(top):
                sys.exit(f"{pin['asset']}: unexpected entry {member.name}; not unpacked")
            member.name = member.name[len(top):]
            archive.extract(member, partial, filter="data")
    (partial / ".sha256").write_text(pin["sha256"])
    shutil.rmtree(target, ignore_errors=True)
    partial.replace(target)
    print(f"Dawn {pins['release']} for {key}: unpacked into {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

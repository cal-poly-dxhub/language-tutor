"""
Container runtime preflight.

The stack has two `DockerImageAsset`s, so a deploy needs a working image builder. The
CDK CLI picks `process.env.CDK_DOCKER ?? "docker"` — so on a Mac with the Docker CLI
installed but no Docker Desktop daemon, `cdk deploy` synthesizes happily and then dies
minutes later with:

    docker build --tag cdkasset-... exited with error code 1
    ERROR: failed to connect to the docker API at unix:///.../docker.sock

That message names the symptom, not the cause. This check runs during synth — including
under a plain `cdk deploy` — and turns it into an immediate, actionable error.

Set CDK_SKIP_RUNTIME_CHECK=1 to bypass (CI with a remote builder, for example).
"""

import os
import shutil
import subprocess

SKIP_ENV = "CDK_SKIP_RUNTIME_CHECK"


def _works(binary: str) -> bool:
    """True if the client can actually reach a daemon, not just exist on PATH."""
    if not shutil.which(binary):
        return False
    try:
        return subprocess.run(
            [binary, "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=30, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def check(*, skip: bool = None) -> None:
    if skip is None:
        skip = os.environ.get(SKIP_ENV) == "1"
    if skip:
        return

    chosen = os.environ.get("CDK_DOCKER")
    if chosen:
        # An explicit choice is respected, but still verified — a stale
        # `export CDK_DOCKER=docker` is otherwise indistinguishable from no setting.
        if _works(chosen):
            return
        raise SystemExit(
            f"\nCDK_DOCKER is set to '{chosen}', but '{chosen} info' failed, so image "
            f"assets cannot be built.\nUnset it and use ./deploy.sh, which picks a "
            f"runtime that works:\n\n    unset CDK_DOCKER && ./deploy.sh\n")

    if _works("docker"):
        return

    hint = ("    ./deploy.sh                     # instead of: npx cdk deploy\n"
            "    ./deploy.sh -c phonemes=false   # skip the slow GPU image\n")
    if shutil.which("finch"):
        raise SystemExit(
            "\nNo usable container builder: the 'docker' CLI is on PATH but its daemon "
            "is not reachable,\nand CDK_DOCKER is not set — so `cdk deploy` would fail "
            "on the image assets.\n\nFinch is installed. Deploy through the wrapper, "
            "which points CDK at it:\n\n" + hint)
    raise SystemExit(
        "\nNo usable container builder found, and the stack builds two container "
        "images.\nInstall Finch (brew install --cask finch && finch vm init) and "
        "deploy with:\n\n" + hint)

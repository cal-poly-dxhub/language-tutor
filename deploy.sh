#!/usr/bin/env bash
#
# CDK wrapper that picks a working container runtime. USE THIS INSTEAD OF `npx cdk`.
#
# The stack builds two container images (the Nova Sonic bridge, and the Wav2Vec2
# SageMaker model), so a deploy needs a working builder. The CDK CLI defaults to
# whatever `docker` binary is on PATH — and on a Mac with the Docker CLI installed but
# no Docker Desktop daemon running, the build fails with:
#
#   docker build --tag cdkasset-... exited with error code 1
#   ERROR: failed to connect to the docker API at unix:///.../docker.sock
#
# If you see `docker build` in that message, cdk was run WITHOUT this wrapper.
#
# Finch is a drop-in for these builds. This script belts-and-braces it two ways:
#   1. exports CDK_DOCKER=finch, which the CDK CLI uses to build image assets; and
#   2. puts ./bin first on PATH, where a `docker` shim forwards to finch — so even a
#      code path that ignores CDK_DOCKER and shells out to `docker` still works.
#
# Usage:
#   ./deploy.sh                             # deploy LanguageTutorStack
#   ./deploy.sh -c phonemes=false           # skip the GPU endpoint (much faster build)
#   ./deploy.sh -c language=french          # teach a different language
#   ./deploy.sh synth                       # any cdk subcommand works
#   CDK_DOCKER=docker ./deploy.sh           # force a runtime yourself

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

start_finch_vm() {
  if [[ "$(finch vm status 2>/dev/null || true)" != *Running* ]]; then
    echo "Finch VM is not running — starting it…" >&2
    if ! finch vm start >&2; then
      echo "Could not start the Finch VM. If this is a fresh install, run: finch vm init" >&2
      exit 1
    fi
  fi
}

pick_runtime() {
  # Respect an explicit choice.
  if [[ -n "${CDK_DOCKER:-}" ]]; then
    [[ "$CDK_DOCKER" == finch ]] && start_finch_vm
    echo "$CDK_DOCKER"
    return
  fi

  if command -v finch >/dev/null 2>&1; then
    start_finch_vm
    echo finch
    return
  fi

  # No finch: fall back to docker only if its daemon actually answers.
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    echo docker
    return
  fi

  echo "No working container runtime found. Install Finch (brew install --cask finch)" >&2
  echo "then run 'finch vm init', or start Docker Desktop." >&2
  exit 1
}

CDK_DOCKER="$(pick_runtime)"
export CDK_DOCKER

# Only shim `docker` when we are actually delegating to finch.
if [[ "$CDK_DOCKER" == finch ]]; then
  export PATH="$HERE/bin:$PATH"
fi

echo "==> container runtime: $CDK_DOCKER  (docker -> $(command -v docker))" >&2

# Node 23+ trips a jsii support warning that is noise here.
export JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION=1

# A bare invocation means "deploy the app"; anything else is passed through verbatim so
# `./deploy.sh synth`, `./deploy.sh diff -c phonemes=false`, etc. all work.
if [[ $# -eq 0 ]]; then
  set -- deploy LanguageTutorStack
elif [[ "$1" == -* ]]; then
  set -- deploy LanguageTutorStack "$@"
fi

exec npx cdk "$@"

#!/usr/bin/env bash
###############################################################################
# Stable Audio 3 — Docker deploy script
#
#   ./deploy.sh build            build the image
#   ./deploy.sh up [--build] [--wait]   start the service (GPU, port 7860)
#   ./deploy.sh down             stop and remove the container
#   ./deploy.sh restart          down + up (keeps --wait/--build flags)
#   ./deploy.sh logs [-f]        follow container logs
#   ./deploy.sh status           container / health / URL
#   ./deploy.sh doctor           check docker, GPU toolchain, disk, image
#   ./deploy.sh clean [--all]    remove the image (--all also drops any
#                                leftover HF cache volume from older setups)
#
# Configuration (env vars, overridable per invocation):
#   IMAGE=stable-audio-3:latest   image tag (weights are baked into it)
#   CONTAINER_NAME=stable-audio-3 container name
#   PORT=7860                     host port
#   WAIT_TIMEOUT=1800             seconds to wait for "healthy" with --wait
#   HF_TOKEN=...                  HF token for the gated SA3 repos; falls
#                                 back to ~/.cache/huggingface/token
#   SA3_VARIANTS=medium           checkpoints the container preloads
#                                 (all | medium | small-music,small-sfx …)
#   PRELOAD_MODELS=0              passed to 'build': skip baking the
#                                 ~17 GB of weights into the image
###############################################################################
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

IMAGE="${IMAGE:-stable-audio-3:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-stable-audio-3}"
PORT="${PORT:-7860}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
PRELOAD_MODELS="${PRELOAD_MODELS:-1}"
# Cache volume from earlier revisions; only 'clean --all' still touches it.
HF_VOLUME="${HF_VOLUME:-stable-audio-3-hf}"
APP_PORT=7860   # port the app listens on inside the container

if [[ -t 1 ]]; then
    C_RESET=$'\033[0m' C_BOLD=$'\033[1m'
    C_BLUE=$'\033[34m' C_GREEN=$'\033[32m' C_YELLOW=$'\033[33m' C_RED=$'\033[31m'
else
    C_RESET='' C_BOLD='' C_BLUE='' C_GREEN='' C_YELLOW='' C_RED=''
fi

info() { printf '%s==>%s %s\n' "$C_BLUE"   "$C_RESET" "$*"; }
ok()   { printf '%s  ok%s %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn() { printf '%swarn%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
die()  { printf '%sfail%s %s\n' "$C_RED"   "$C_RESET" "$*" >&2; exit 1; }

usage() { sed -n '3,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

require_docker() {
    command -v docker >/dev/null 2>&1 || die "docker not found in PATH."
    docker info >/dev/null 2>&1 || die "docker daemon is not reachable (start Docker, or fix permissions)."
}

# The SA3 repos are gated: $HF_TOKEN, else the token huggingface-cli stored
# on the host. Used both as a build secret and as a runtime env var.
hf_token() {
    if [[ -n "${HF_TOKEN:-}" ]]; then
        printf '%s' "$HF_TOKEN"
        return 0
    fi
    local f="${HOME}/.cache/huggingface/token"
    if [[ -r "$f" ]]; then
        tr -d '[:space:]' < "$f"
    fi
    return 0
}

port_busy() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltnH "sport = :$PORT" 2>/dev/null | grep -q . && return 0
    fi
    return 1
}

# Cheap GPU probe: runs torch inside the image and asserts CUDA is visible.
gpu_probe() {
    local out
    if ! out=$(docker run --rm --gpus all "$IMAGE" python -c \
            'import torch
assert torch.cuda.is_available(), "torch: CUDA unavailable"
print(torch.cuda.get_device_name(0))' 2>&1); then
        warn "GPU probe failed:"
        printf '%s\n' "$out" | tail -n 6 >&2
        die "cannot use the GPU from Docker (run './deploy.sh doctor' for details)."
    fi
    ok "GPU: ${out//$'\n'/}"
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_build() {
    require_docker
    local extra=()
    [[ "${1:-}" == "--no-cache" ]] && extra+=(--no-cache)

    local tok secrets=() build_env=()
    tok="$(hf_token)"
    if [[ "$PRELOAD_MODELS" == "0" ]]; then
        build_env+=(PRELOAD_MODELS=0)
        info "building $IMAGE (PRELOAD_MODELS=0: weights download at first boot)"
    elif [[ -n "$tok" ]]; then
        secrets+=(--secret id=hf_token,env=HF_TOKEN)
        build_env+=("HF_TOKEN=$tok")
        info "building $IMAGE (baking the three gated checkpoints, ~17 GB)"
    else
        warn "no HF token found (\$HF_TOKEN or ~/.cache/huggingface/token)."
        warn "the SA3 repos are gated — the preload step will fail unless you"
        warn "accept the license and set a token, or build with PRELOAD_MODELS=0."
    fi

    env "${build_env[@]}" docker build "${secrets[@]}" "${extra[@]}" -t "$IMAGE" "$SCRIPT_DIR"
    ok "built $IMAGE"
}

cmd_up() {
    require_docker
    local do_build=0 do_wait=0
    for arg in "$@"; do
        case "$arg" in
            --build) do_build=1 ;;
            --wait)  do_wait=1 ;;
            -h|--help) usage; exit 0 ;;
            *) die "unknown 'up' flag: $arg" ;;
        esac
    done

    if [[ $do_build -eq 1 ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        cmd_build
    fi

    gpu_probe

    if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        info "removing existing container '$CONTAINER_NAME'"
        docker rm -f "$CONTAINER_NAME" >/dev/null
    fi
    if port_busy; then
        warn "host port $PORT is already in use — the service may not be reachable."
    fi

    local tok; tok="$(hf_token)"
    local -a env_args=()
    [[ -n "$tok" ]] && env_args+=(-e "HF_TOKEN=$tok")
    # Which checkpoints the container preloads (image default: medium).
    [[ -n "${SA3_VARIANTS:-}" ]] && env_args+=(-e "SA3_VARIANTS=$SA3_VARIANTS")

    info "starting '$CONTAINER_NAME' (port $PORT -> $APP_PORT, weights baked into the image)"
    docker run -d \
        --name "$CONTAINER_NAME" \
        --init \
        --restart unless-stopped \
        --gpus all \
        -p "$PORT:$APP_PORT" \
        "${env_args[@]}" \
        "$IMAGE" >/dev/null

    ok "container started"
    printf '       %surl%s       http://localhost:%s\n' "$C_BOLD" "$C_RESET" "$PORT"
    printf '       %slogs%s      ./deploy.sh logs -f\n' "$C_BOLD" "$C_RESET"

    if [[ $do_wait -eq 1 ]]; then
        cmd_wait_healthy
    else
        warn "startup loads all three checkpoints from the image (~17 GB);"
        warn "the app is only ready once the container reports 'healthy'."
    fi
}

cmd_wait_healthy() {
    info "waiting up to ${WAIT_TIMEOUT}s for healthy (loading models) …"
    local waited=0 last=""
    while (( waited < WAIT_TIMEOUT )); do
        local state health
        state=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo missing)
        health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' \
                     "$CONTAINER_NAME" 2>/dev/null || echo missing)
        [[ "$state" != "$last" ]] && { info "container: $state, health: $health"; last="$state"; }
        if [[ "$state" == "exited" || "$state" == "dead" || "$state" == "missing" ]]; then
            docker logs --tail 40 "$CONTAINER_NAME" >&2 || true
            die "container is $state before becoming healthy."
        fi
        if [[ "$health" == "healthy" ]]; then
            ok "healthy — http://localhost:$PORT"
            return 0
        fi
        sleep 5; waited=$((waited + 5))
    done
    warn "still not healthy after ${WAIT_TIMEOUT}s — check './deploy.sh logs'."
    docker inspect -f '{{json .State.Health}}' "$CONTAINER_NAME" 2>/dev/null || true
    return 1
}

cmd_down() {
    require_docker
    if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        info "stopping '$CONTAINER_NAME'"
        docker rm -f "$CONTAINER_NAME" >/dev/null
        ok "stopped"
    else
        warn "container '$CONTAINER_NAME' does not exist"
    fi
}

cmd_logs() {
    require_docker
    docker logs "$@" "$CONTAINER_NAME"
}

cmd_status() {
    require_docker
    if ! docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        warn "container '$CONTAINER_NAME' is not created (run './deploy.sh up')"
        return 0
    fi
    docker ps -a --filter "name=^${CONTAINER_NAME}$" \
        --format '  name    {{.Names}}  status={{.Status}}  image={{.Image}}'
    local health
    health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}no healthcheck{{end}}' \
                 "$CONTAINER_NAME" 2>/dev/null || echo unknown)
    printf '  health  %s\n  url     http://localhost:%s\n' "$health" "$PORT"
    docker stats --no-stream --format '  stats   cpu={{.CPUPerc}} mem={{.MemUsage}} gpu={{.GPUPerc}}' \
        "$CONTAINER_NAME" 2>/dev/null || true
}

cmd_doctor() {
    info "checking docker"
    command -v docker >/dev/null 2>&1 || die "docker not found in PATH."
    docker info >/dev/null 2>&1 || die "docker daemon is not reachable."
    ok "docker $(docker --version | awk '{print $3}') / compose $(docker compose version --short 2>/dev/null || echo n/a)"

    info "checking nvidia container toolkit"
    if docker info --format '{{range .Runtimes}}{{.Key}} {{end}}' 2>/dev/null | grep -qw nvidia; then
        ok "nvidia runtime registered with dockerd"
    else
        warn "no 'nvidia' runtime registered (CDI setups are fine — the probe below decides)."
    fi

    info "checking host GPU driver"
    if command -v nvidia-smi >/dev/null 2>&1; then
        ok "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n1)"
    else
        warn "nvidia-smi not found on the host"
    fi

    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        info "probing GPU from inside the image"
        gpu_probe
    else
        warn "image $IMAGE not built yet — skipping in-container GPU probe."
    fi

    info "checking disk space"
    local free
    free=$(df -BG --output=avail "$SCRIPT_DIR" | tail -n1 | tr -dc '0-9')
    if (( free >= 40 )); then
        ok "${free}G free — image with baked weights needs ~40G"
    else
        warn "${free}G free — building the image (~40G with weights) may not fit."
    fi
}

cmd_clean() {
    require_docker
    cmd_down
    if [[ "${1:-}" == "--all" ]]; then
        info "removing leftover cache volume '$HF_VOLUME'"
        docker volume rm -f "$HF_VOLUME" >/dev/null 2>&1 || true
    fi
    info "removing image $IMAGE"
    docker rmi "$IMAGE" >/dev/null 2>&1 || warn "image $IMAGE not found"
    ok "cleaned"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

main() {
    local cmd="${1:-help}"; shift || true
    case "$cmd" in
        build)   cmd_build "$@" ;;
        up)      cmd_up "$@" ;;
        down|stop) cmd_down "$@" ;;
        restart) cmd_down; cmd_up "$@" ;;
        logs)    cmd_logs "$@" ;;
        status)  cmd_status "$@" ;;
        doctor)  cmd_doctor "$@" ;;
        clean)   cmd_clean "$@" ;;
        help|-h|--help) usage ;;
        *) usage; die "unknown command: $cmd" ;;
    esac
}

main "$@"

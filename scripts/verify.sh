#!/usr/bin/env bash
# Run local checks that are useful before handing work to review.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STRICT="${VERIFY_STRICT:-0}"
SECURITY_STRICT="${VERIFY_SECURITY_STRICT:-0}"
FAILURES=0
SKIPS=0

run_check() {
    local name="$1"
    shift

    echo "==> ${name}"
    if "$@"; then
        echo "PASS: ${name}"
    else
        local status=$?
        echo "FAIL: ${name} (exit ${status})"
        FAILURES=$((FAILURES + 1))
    fi
    echo
}

skip_check() {
    local name="$1"
    local reason="$2"

    echo "SKIP: ${name} - ${reason}"
    echo
    SKIPS=$((SKIPS + 1))
    if [[ "$STRICT" == "1" ]]; then
        FAILURES=$((FAILURES + 1))
    fi
}

run_advisory_check() {
    local name="$1"
    shift

    echo "==> ${name}"
    if "$@"; then
        echo "PASS: ${name}"
    else
        local status=$?
        if [[ "$SECURITY_STRICT" == "1" ]]; then
            echo "FAIL: ${name} (exit ${status})"
            FAILURES=$((FAILURES + 1))
        else
            echo "ADVISORY: ${name} reported findings or could not complete (exit ${status})"
            echo "Set VERIFY_SECURITY_STRICT=1 to make this advisory check fail the run."
        fi
    fi
    echo
}

has_command() {
    command -v "$1" >/dev/null 2>&1
}

run_check "python compileall" python3 -m compileall app tests

if python3 -m pytest --version >/dev/null 2>&1; then
    run_check "pytest" python3 -m pytest tests
else
    skip_check "pytest" "pytest is not installed; install requirements-dev.txt"
fi

if has_command npm; then
    if [[ -d frontend/node_modules ]]; then
        run_check "frontend build" npm run build --prefix frontend
    else
        skip_check "frontend build" "frontend/node_modules is missing; run npm ci --prefix frontend"
    fi
    run_check "frontend production npm audit" npm audit --omit=dev --audit-level=high --prefix frontend
else
    skip_check "frontend checks" "npm is not installed"
fi

if python3 -m pip_audit --version >/dev/null 2>&1; then
    run_advisory_check "python dependency audit" python3 -m pip_audit -r requirements.txt
else
    skip_check "python dependency audit" "pip-audit is not installed; install requirements-dev.txt"
fi

if python3 -m bandit --version >/dev/null 2>&1; then
    run_advisory_check "bandit app scan" python3 -m bandit -q -r app
else
    skip_check "bandit app scan" "bandit is not installed; install requirements-dev.txt"
fi

if has_command git && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    run_check "git diff whitespace" git diff --check
    run_check "git staged diff whitespace" git diff --cached --check
fi

echo "Verification summary: ${FAILURES} failure(s), ${SKIPS} skip(s)"

if [[ "$FAILURES" -ne 0 ]]; then
    exit 1
fi

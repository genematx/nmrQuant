#!/bin/bash
set -e
set -x

case "$1" in
    bash)
        bash #-c "${@:2}"
    ;;
    test)
        coverage erase
        coverage run --branch -m pytest "${@:2}"
        coverage xml -i -o coverage/coverage.xml
    ;;
    fmt)
        isort src tests "${@:2:2}"
        black ${PWD} "${@:4}"
    ;;
    lint)
        flake8 ${PWD}
        mypy src || true
    ;;
    *)
        exec "$@"
    ;;
esac
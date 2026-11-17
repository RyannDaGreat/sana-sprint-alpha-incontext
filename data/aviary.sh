#!/bin/sh
# Runs Aviary (Bluejay, Sparrow, Albatross) from the checkout in data/Aviary (corp/Aviary, branch bluejay2). Every script calls this file
# setup.sh clones the checkout and builds it into the one wheel in data/Aviary/.wheel; run setup.sh again after a git pull there
# --python 3.13: Aviary needs 3.13. Netflix's PyPI is searched first, as install_local.sh does: public PyPI's metatron is an unrelated project
exec uvx --python 3.13 --index https://pypi.netflix.net/simple --default-index https://pypi.org/simple \
    --from "$(ls "$(dirname "$0")"/Aviary/.wheel/*.whl)" aviary "$@"

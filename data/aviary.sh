#!/bin/sh
# Runs nflx-aviary (Bluejay, Sparrow, Albatross) at the pinned version. Every script calls this file; change the pin here only
# --python 3.13: nflx-aviary needs 3.13
exec uvx --python 3.13 --from 'nflx-aviary>=0.0.53' nflx-aviary "$@"

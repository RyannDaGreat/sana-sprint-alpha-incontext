#!/usr/bin/env fish
# Opens the BlueJay tables (curated MAGICK data, eval generations) in Sparrow, the browser viewer. Extra arguments go to Sparrow
# Usage: ./run_sparrow.fish [--port P]

set PORT 26400  # Far from the BlueJay database (23947) and the web UI (41730)
exec fish (status dirname)/data/run_sparrow.fish --port $PORT $argv

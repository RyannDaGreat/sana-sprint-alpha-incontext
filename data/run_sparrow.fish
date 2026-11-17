#!/usr/bin/env fish
# Opens this project in Sparrow, the browser viewer. Extra arguments go to Sparrow

cd (status dirname); or exit
./aviary.sh bluejay initialize project.dtab; or exit  # By convention, every launch script initializes first
./aviary.sh sparrow project.dtab $argv

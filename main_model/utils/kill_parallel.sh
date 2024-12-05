#!/bin/bash

# Get a list of PIDs for processes with the name 'graphgen_main'
PIDS=$(pgrep -f graphgen_main)

# Check if we got any PIDs
if [ -z "$PIDS" ]; then
  echo "No 'graphgen_main' processes found to kill."
else
  # Loop through all PIDs and kill them with SIGTERM (-15)
  for PID in $PIDS; do
    echo "Killing 'graphgen_main' process with PID: $PID"
    kill -15 $PID
  done
fi

#!/bin/sh

for i in $(seq 1 $1)
do
    nohup python -u graphgen_main.py $i $1 $2 $3 $4 &
done

#!/bin/sh

for i in $(seq 1 $1)
do
    for j in $(seq 1 $2)
    do
        nohup python -u graphgen_main.py $i $j $1 $2 $3 $4 $5 $6 &
    done
done

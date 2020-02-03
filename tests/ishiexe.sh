#!/bin/bash

# get pid
pid=${1}

# include pyeee in PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:${PWD}

# make individual run directory
mkdir tmp.${pid}

# run in individual directory
cp tests/ishiexe.py tmp.${pid}/
mv params.txt.${pid} tmp.${pid}/params.txt
cd tmp.${pid}
python3 ishiexe.py

# make output available to pyeee
mv obj.txt ../obj.txt.${pid}

# clean up
cd ..
rm -r tmp.${pid}

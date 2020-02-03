#!/bin/bash

# get pid
pid=${1}

# make individual run directory
mkdir tmp.${pid}

# run in individual directory
cp tests/ishiexe.py tmp.${pid}/
mv params.txt.${pid} tmp.${pid}/params.txt
cd tmp.${pid}
ln -s ../pyeee/std_io.py 
ln -s ../pyeee/sa_test_functions.py 
python3 ishiexe.py

# make output available to pyeee
mv obj.txt ../obj.txt.${pid}

# clean up
cd ..
rm -r tmp.${pid}

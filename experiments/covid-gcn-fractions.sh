#!/bin/bash

for fraction in 0.01 0.02 0.4 0.6
do
echo '******************** new seed ***************************'
echo $seed
    for seed in 1 2 3 4 5
    do
    THEANO_FLAGS='device=cuda0,floatX=float32' nice -n 9 python -u gcnmain.py -hid 500 400 300 200 129 -bucket 50 -batch 500 -d ./datasets/covid/  -mindf 14 -reg 0.0 -dropout 0.0 -cel 5 -highway -seed $seed -lblfraction $fraction -builddata -silent #-maxdown 50
    done
done
exit 0

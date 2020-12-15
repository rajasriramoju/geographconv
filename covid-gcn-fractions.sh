#!/bin/bash

#for seed in 1 2 3 4 5
#for fraction in 0.01 0.02 0.4 0.6
for fraction in 0.6 0.4
do
echo '******************** new seed ***************************'
echo $seed
    #for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
    for seed in 2 3 4 5
    do
    THEANO_FLAGS='device=cuda0,floatX=float32' nice -n 9 python -u gcnmain.py -hid 500 400 300 -bucket 120 -batch 500 -d ./datasets/covid/  -mindf 14 -reg 0.0 -dropout 0.5 -cel 4 -highway -seed $seed -lblfraction $fraction -builddata -silent -maxdown 50


    done
done
exit 0

#!/bin/sh

for run in 1 2 3 4
do
	for grid in 512 1024
	do
		for eps in 0.01 0.001
		do
			for proc in 32 64 128 256 512
			do
				mpisubmit.bg -n $proc -w 24:00:00 ./par -- $grid $eps 
			done
		done
	done
done

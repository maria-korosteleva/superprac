#!/bin/sh

	for grid in 512 1024
	do
		for eps in 0.01 0.001
		do
				mpirun -np 1 ./par $grid $eps "res_grid/grid_$grid_$eps_$proc_$run.txt"
			done
		done

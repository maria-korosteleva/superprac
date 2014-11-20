#!/bin/sh

for grid in res_grid/*; 
do 
	python plot.py $grid
done


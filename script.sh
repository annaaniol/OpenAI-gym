#!/bin/bash
for alpha in 0.1 0.2 0.5
do
	for gamma in 0.5 0.7 0.9
	do
		for eps in 0.01 0.05 0.1
		do
			for buckets in 2 3 4 5 6
			do
				python3 balance.py $alpha $gamma $eps $buckets -1 False
				python3 plot.py $alpha $gamma $eps $buckets
			done
		done
	done
done

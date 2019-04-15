#!/bin/bash
for alpha in 0.1
do
	for gamma in 0.5 0.7 0.9
	do
		for eps in 0.01 0.05 0.1
		do
			for buckets in 4 5 6
			do
				python3 balance.py $alpha $gamma $eps $buckets -1 False 5
				python3 sarsa.py $alpha $gamma $eps $buckets -1 False 5
				python3 plot.py $alpha $gamma $eps $buckets
			done
		done
	done
done

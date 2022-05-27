#!/bin/bash

# Exponential landscape
./run.sh 50 exponential-growth exponential-growth "exponential_correlated" &          # Correlated case
./run.sh 50 exponential-growth exponential-decay "exponential_anticorrelated" &       # Anti-correlated case
./run.sh 50 exponential-growth random-uniform "exponential_uncorrelated" &            # Uncorrelated case

# Two-basin landscape
./run.sh 50 two-basin-a-better two-basin-a-better "twobasin_correlated" &          # Correlated case
./run.sh 50 two-basin-a-better two-basin-b-better "twobasin_anticorrelated" &      # Anti-correlated case
./run.sh 50 two-basin-a-better random-uniform "twobasin_uncorrelated" &            # Uncorrelated case
#!/usr/bin/env python3
"""
Generates a FAKE file in MovieLens-1M format (for testing the code when the real
dataset is not available).   Usage: python3 make_synthetic.py out.dat USERS ITEMS
"""
import sys, random
out, U, I = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rng = random.Random(1)
with open(out, "w") as f:
    for u in range(1, U + 1):
        for m in rng.sample(range(1, I + 1), min(I, rng.randint(20, 120))):
            f.write(f"{u}::{m}::{rng.randint(1, 5)}::978300000\n")

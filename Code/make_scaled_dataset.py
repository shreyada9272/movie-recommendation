##
# # Creates scaled versions of a ratings dataset for scalability
# experiments.

# The generated datasets help evaluate how execution time changes
# as the number of users and items increases.
##

#!/usr/bin/env python3
"""
Create a LARGER dataset from MovieLens 1M by replicating the users K times.
Copy c gets user ids  u + c*6040.  10% of ratings in the copies are changed by +-1
(fixed random seed) so the copies are not exact duplicates.

Usage:  python3 make_scaled_dataset.py ml-1m/ratings.dat ratings_x4.dat 4
"""
import sys, random

src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
rng = random.Random(42)
rows = [l.strip().split("::") for l in open(src)]
max_user = max(int(r[0]) for r in rows)

with open(dst, "w") as out:
    for c in range(k):
        for u, m, r, t in rows:
            r = int(r)
            if c > 0 and rng.random() < 0.10:
                r = min(5, max(1, r + rng.choice((-1, 1))))
            out.write(f"{int(u) + c * max_user}::{m}::{r}::{t}\n")
print(f"wrote {dst}: {k * len(rows)} ratings, {k * max_user} users")


#!/usr/bin/env python3
"""
Runs the full experiment grid and VERIFIES that sequential and OpenMP outputs are identical.

  python3 run_benchmark.py --data ml-1m/ratings.dat
  python3 run_benchmark.py --data ml-1m/ratings.dat --users 1000 6040 --items 1000 3952 --threads 1 2 4 8 --reps 3

Writes results.csv (one row per run).
"""
import argparse, csv, filecmp, os, subprocess, sys, tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--users", type=int, nargs="+", default=[1000, 2000, 4000, 6040])
ap.add_argument("--items", type=int, nargs="+", default=[500, 1000, 2000, 3952])
ap.add_argument("--threads", type=int, nargs="+", default=None)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", default="results.csv")
a = ap.parse_args()

if a.threads is None:
    cores, t, a.threads = os.cpu_count() or 1, 1, []
    while t <= cores:
        a.threads.append(t); t *= 2
    if a.threads[-1] != cores: a.threads.append(cores)
print("threads:", a.threads)

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0: sys.exit(p.stderr)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")][0].split(",")
    return dict(version=line[1], users=int(line[2]), items=int(line[3]), ratings=int(line[4]),
                threads=int(line[5]), load=float(line[6]), sim=float(line[7]),
                rec=float(line[8]), total=float(line[9]), checksum=line[10])

tmp = tempfile.mkdtemp()
rows, all_ok = [], True
for U in a.users:
    for I in a.items:
        for rep in range(a.reps):
            ref = os.path.join(tmp, "seq.txt")
            s = run(["./recommender_seq", a.data, str(U), str(I), ref]); s["rep"] = rep; s["match"] = "ref"
            rows.append(s)
            line = f"U={U:5d} I={I:5d} rep={rep} seq={s['total']:8.3f}s |"
            for T in a.threads:
                o_path = os.path.join(tmp, f"omp{T}.txt")
                o = run(["./recommender_omp", a.data, str(U), str(I), str(T), o_path]); o["rep"] = rep
                same = filecmp.cmp(ref, o_path, shallow=False) and o["checksum"] == s["checksum"]
                o["match"] = "YES" if same else "NO"; all_ok &= same
                rows.append(o)
                line += f" T{T}={o['total']:.3f}s{'' if same else ' MISMATCH!'}"
            print(line, flush=True)

with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nwritten", a.out)
print("VERIFICATION:", "ALL OpenMP outputs are IDENTICAL to the sequential output" if all_ok
      else "!!! MISMATCH FOUND !!!")

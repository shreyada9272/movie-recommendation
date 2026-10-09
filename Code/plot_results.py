# Reads benchmark results and generates graphs to compare the
# performance of the sequential and OpenMP parallel implementations.

# The plots help analyze execution time, speedup, parallel efficiency,
# and scalability as the workload or thread count changes.

#!/usr/bin/env python3

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Read experimental results
# ---------------------------------------------------------

df = pd.read_csv("results.csv")


# ---------------------------------------------------------
# Calculate speedup and efficiency
# ---------------------------------------------------------

seq = (
    df[df["version"] == "seq"]
    .groupby(["users", "items"])["total"]
    .mean()
    .rename("t_seq")
)

omp = (
    df[df["version"] == "omp"]
    .groupby(["users", "items", "threads"])["total"]
    .mean()
    .rename("t_omp")
    .reset_index()
)

summary = omp.merge(
    seq.reset_index(),
    on=["users", "items"]
)

summary["speedup"] = summary["t_seq"] / summary["t_omp"]
summary["efficiency"] = summary["speedup"] / summary["threads"]


# ---------------------------------------------------------
# Save summary table
# ---------------------------------------------------------

summary.to_csv("summary.csv", index=False)

with open("summary.md", "w") as f:
    f.write("# Performance Summary\n\n")
    f.write(summary.round(3).to_markdown(index=False))
    f.write("\n\nVerification: ")
    
    if (df["match"] != "NO").all():
        f.write("ALL CHECKSUMS MATCH\n")
    else:
        f.write("MISMATCH FOUND\n")


# ---------------------------------------------------------
# Graph 1: Speedup vs Threads
# ---------------------------------------------------------

data = summary[
    (summary["users"] == 6040) &
    (summary["items"] == 3952)
].sort_values("threads")

plt.figure(figsize=(7, 5))

plt.plot(
    data["threads"],
    data["speedup"],
    marker="o",
    linewidth=2,
    label="Measured speedup"
)

plt.plot(
    data["threads"],
    data["threads"],
    linestyle="--",
    label="Ideal speedup"
)

plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Number of Threads")
plt.xticks(data["threads"])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("speedup_vs_threads.png", dpi=200)
plt.close()


# ---------------------------------------------------------
# Graph 2: Parallel Efficiency vs Threads
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(
    data["threads"],
    data["efficiency"] * 100,
    marker="o",
    linewidth=2
)

plt.axhline(
    100,
    linestyle="--",
    label="Ideal efficiency"
)

plt.xlabel("Number of Threads")
plt.ylabel("Parallel Efficiency (%)")
plt.title("Parallel Efficiency vs Number of Threads")
plt.xticks(data["threads"])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("efficiency_vs_threads.png", dpi=200)
plt.close()


# ---------------------------------------------------------
# Graph 3: Execution Time vs Number of Users
# ---------------------------------------------------------

# Sequential results
seq_users = (
    df[df["version"] == "seq"]
    .groupby("users")["total"]
    .mean()
    .sort_index()
)

# OpenMP results using 4 threads
omp4_users = (
    df[
        (df["version"] == "omp") &
        (df["threads"] == 4)
    ]
    .groupby("users")["total"]
    .mean()
    .sort_index()
)

plt.figure(figsize=(7, 5))

plt.plot(
    seq_users.index,
    seq_users.values,
    marker="o",
    linewidth=2,
    label="Sequential"
)

plt.plot(
    omp4_users.index,
    omp4_users.values,
    marker="s",
    linewidth=2,
    label="OpenMP (4 threads)"
)

plt.xlabel("Number of Users")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time vs Number of Users")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("time_vs_users.png", dpi=200)
plt.close()


# ---------------------------------------------------------
# Graph 4: Execution Time vs Threads
# ---------------------------------------------------------

omp_threads = (
    df[
        (df["version"] == "omp") &
        (df["users"] == 6040) &
        (df["items"] == 3952)
    ]
    .sort_values("threads")
)

plt.figure(figsize=(7, 5))

plt.plot(
    omp_threads["threads"],
    omp_threads["total"],
    marker="o",
    linewidth=2
)

plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (seconds)")
plt.title("OpenMP Execution Time vs Number of Threads")
plt.xticks(omp_threads["threads"])
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("execution_time_vs_threads.png", dpi=200)
plt.close()


# ---------------------------------------------------------
# Print final summary
# ---------------------------------------------------------

print("\nPerformance Summary")
print("===================")

print(summary.round(3).to_string(index=False))

print("\nGenerated graphs:")
print("1. speedup_vs_threads.png")
print("2. efficiency_vs_threads.png")
print("3. time_vs_users.png")
print("4. execution_time_vs_threads.png")

print("\nVerification:")
if (df["match"] != "NO").all():
    print("ALL CHECKSUMS MATCH")
else:
    print("CHECKSUM MISMATCH FOUND")
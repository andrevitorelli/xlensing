"""Timing benchmark for the main xlensing operations.

usage: python scripts/benchmark.py

Times (single CPU core, wall clock, median of repeated runs):
  1. NFW_Delta_Sigma model (centred + miscentred NFW + two-halo) for many clusters;
  2. per-galaxy lensing quantities (metacal_cluster_lensing) for one cluster;
  3. a radially binned profile with 200 bootstrap resamples (bootstrap_signal);
  4. exact triaxial NFW shear injection per source galaxy (apply_triaxial_NFW_shear_region).
"""
import time

import numpy as np

from xlensing import data, model, sims


def timeit(fn, repeat=5):
    fn()  # warm-up (lookup tables, caches)
    t = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def main():
    rng = np.random.default_rng(0)

    # 1. model evaluation
    radii = np.logspace(-0.8, 0.6, 16)
    for n in (1_000, 16_000, 100_000):
        M = 10 ** rng.uniform(14, 15, n)
        c = rng.uniform(3, 8, n)
        z = rng.uniform(0.1, 0.7, n)
        f = np.full(n, 0.3)
        s = np.full(n, 0.4)
        b = np.zeros(n)
        t = timeit(lambda: model.NFW_Delta_Sigma(M, c, z, f, s, b, radii))
        print(f"NFW_Delta_Sigma: {n:7d} clusters x 16 radii  {t*1e3:8.1f} ms  ({t/n*1e6:.2f} us per cluster)")

    # 2. per-galaxy measurement for one cluster
    ngal = 100_000
    zc = 0.3
    width = 0.01  # rad
    ra = rng.uniform(-width, width, ngal)
    dec = rng.uniform(-width, width, ngal)
    zs = rng.uniform(0.6, 1.5, ngal)
    e1 = rng.normal(0, 0.27, ngal)
    e2 = rng.normal(0, 0.27, ngal)
    w = np.ones(ngal)
    one = np.ones(ngal)
    zero = np.zeros(ngal)
    sources = np.vstack([ra, dec, zs, e1, e2, w, one, zero, zero, one])  # (10, N): one row per column
    cluster = (0.0, 0.0, zc)
    t = timeit(lambda: data.metacal_cluster_lensing(cluster, sources, 5.0))
    print(f"metacal_cluster_lensing: {ngal} sources  {t*1e3:8.1f} ms  ({t/ngal*1e9:.0f} ns per source)")

    # 3. binned profile + bootstrap covariance
    res = data.metacal_cluster_lensing(cluster, sources, 5.0)
    stake = np.vstack([res["Critical Density"], res["Tangential Shear"], res["Cross Shear"],
                       res["Weights"], res["Radial Distance"], res["Mult. Bias"]])
    edges = np.logspace(-0.9, 0.7, 17)
    bins = np.array([[edges[i], edges[i + 1]] for i in range(16)])
    t = timeit(lambda: data.bootstrap_signal(stake, bins, Nboot=200))
    print(f"bootstrap_signal: {stake.shape[1]} sources, 16 bins, 200 resamples  {t*1e3:8.1f} ms")

    # 4. exact triaxial shear injection (the method used for simulated catalogues)
    nsrc = 40_000
    gals = np.column_stack([rng.uniform(-width, width, nsrc), rng.uniform(-width, width, nsrc),
                            rng.uniform(0.6, 1.5, nsrc), rng.normal(0, 0.27, nsrc), rng.normal(0, 0.27, nsrc)])
    rv = sims.r_vir(zc, 2e14)
    clus = [0.0, 0.0, zc, 2e14, 4.0, rv, 0.7 * rv, 0.5 * rv, 0.6, 0.3]   # RA, DEC, z, M200, c200, a, b, c, theta_los, phi_los
    t = timeit(lambda: sims.apply_triaxial_NFW_shear_region(clus, gals), repeat=3)
    print(f"apply_triaxial_NFW_shear_region (exact): {nsrc} sources  {t*1e3:8.1f} ms  ({t/nsrc*1e6:.2f} us per source)")


if __name__ == "__main__":
    main()

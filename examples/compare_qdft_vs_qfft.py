"""compare_qdft_vs_qfft.py — compare direct vs fast QDFT outputs and timing.

Demonstrates that the direct O(N²) reference implementation (qdft) and the
fast O(N log N) FFT-based implementation (qfft) produce numerically
identical results, and shows the runtime difference for a modest signal
length.

Usage::

    python examples/compare_qdft_vs_qfft.py
"""

import time

import numpy as np

from qsp_fft import qdft, qfft


def main() -> None:
    rng = np.random.default_rng(42)

    N = 64
    q = rng.standard_normal((N, 4))
    u = np.array([1.0, 0.0, 0.0])

    print(f"Signal length : N = {N}")
    print(f"Analysis axis : {u}")
    print()

    # Direct (reference)
    t0 = time.perf_counter()
    Q_direct = qdft(q, u)
    t_direct = time.perf_counter() - t0

    # Fast
    t0 = time.perf_counter()
    Q_fast = qfft(q, u)
    t_fast = time.perf_counter() - t0

    # Agreement
    max_diff = float(np.max(np.abs(Q_direct - Q_fast)))
    print(f"Max component-wise difference : {max_diff:.2e}")
    print(f"Outputs agree (atol=1e-8)     : {max_diff < 1e-8}")
    print()
    print(f"qdft  runtime : {t_direct * 1e3:.2f} ms")
    print(f"qfft  runtime : {t_fast * 1e3:.2f} ms")
    if t_fast > 0:
        print(f"Speedup       : {t_direct / t_fast:.1f}x")


if __name__ == "__main__":
    main()

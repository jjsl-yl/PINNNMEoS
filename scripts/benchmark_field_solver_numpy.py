#!/usr/bin/env python3
"""Single-core NumPy benchmark for the variable-10 FSU-delta PINN."""

import argparse
import json
import math
import os
import time
import zipfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from scipy.optimize import fsolve
from scipy.special import expit


HBARC = 197.326968
PI = math.pi
INPUT_PARAM_NAMES = [
    "g_sigma", "g_omega", "g_rho", "g_a0", "b6",
    "g2", "g3", "c3", "c4", "m_sigma",
]
PARAMS = {
    "g_sigma": -9.770109960655295,
    "g_omega": 12.884663945566437,
    "g_rho": 7.008954689269367,
    "g_a0": -6.7423962928532575,
    "b6": 187.942028466664,
    "g2": 1644.1640459214912,
    "g3": 5.983086268779212,
    "c3": 161.82097238231964,
    "c4": 203.68789171819614,
    "m_sigma": 483.64105174950686,
    "m_omega": 782.5,
    "m_rho": 763.0,
    "m_a0": 980.0,
    "M_N": 939.0,
}


class NumpyMLP:
    def __init__(self, checkpoint):
        shapes = [
            (256, 12), (256,),
            (256, 256), (256,),
            (256, 256), (256,),
            (256, 256), (256,),
            (256, 256), (256,),
            (256, 256), (256,),
            (4, 256), (4,),
        ]
        with np.load(checkpoint, allow_pickle=False) as archive:
            arrays=[archive[f'array_{index}'].copy() for index in range(len(shapes))]
            if any(array.shape!=shape for array,shape in zip(arrays,shapes)):
                raise ValueError('Expected six hidden layers of width 256 and 12 inputs')
            self.scale=archive['input_scale'].copy()
            self.step=int(archive['step'])
            self.input_parameters=archive['input_parameters'].copy()
        self.layers = [(arrays[i], arrays[i + 1]) for i in range(0, 12, 2)]
        self.output = (arrays[12], arrays[13])

    @staticmethod
    def silu(x):
        return x * expit(x)

    def __call__(self, x):
        value = np.asarray(x, dtype=np.float32)
        for weight, bias in self.layers:
            value = self.silu(value @ weight.T + bias)
        weight, bias = self.output
        value = value @ weight.T + bias
        fields = np.empty_like(value)
        fields[:, 0] = 120.0 * expit(value[:, 0])
        fields[:, 1] = 120.0 * expit(value[:, 1])
        fields[:, 2] = 40.0 * np.tanh(value[:, 2])
        fields[:, 3] = 40.0 * np.tanh(value[:, 3])
        return fields


def scalar_density(kf, mass):
    mass = max(float(mass), 1.0e-8)
    t = kf / mass
    return mass**3 / PI**2 * 0.5 * (
        t * math.sqrt(1.0 + t * t) - math.asinh(t)
    )


def residual(fields, rho_p_fm3, rho_n_fm3):
    p = PARAMS
    sigma, omega, rho, a0 = fields
    rho_p = rho_p_fm3 * HBARC**3
    rho_n = rho_n_fm3 * HBARC**3
    kfp = (3.0 * PI**2 * rho_p) ** (1.0 / 3.0)
    kfn = (3.0 * PI**2 * rho_n) ** (1.0 / 3.0)
    mp = p["M_N"] + p["g_sigma"] * sigma + p["g_a0"] * a0
    mn = p["M_N"] + p["g_sigma"] * sigma - p["g_a0"] * a0
    rho_sp = scalar_density(kfp, mp)
    rho_sn = scalar_density(kfn, mn)
    return np.asarray([
        p["m_sigma"]**2 * sigma + p["g2"] * sigma**2
        + p["g3"] * sigma**3 - p["b6"] * a0**2 * sigma
        + p["g_sigma"] * (rho_sp + rho_sn),
        p["m_omega"]**2 * omega + p["c3"] * omega**3
        + p["c4"] * omega * rho**2 - p["g_omega"] * (rho_p + rho_n),
        p["m_rho"]**2 * rho + p["c4"] * omega**2 * rho
        - p["g_rho"] * (rho_p - rho_n),
        p["m_a0"]**2 * a0 - p["b6"] * sigma**2 * a0
        + p["g_a0"] * (rho_sp - rho_sn),
    ], dtype=float)


def build_grid(n_density, n_fraction):
    densities = np.linspace(0.005, 1.2, n_density)
    fractions = np.linspace(0.02, 0.50, n_fraction)
    return np.asarray([
        (rho_b * yp, rho_b * (1.0 - yp))
        for yp in fractions
        for rho_b in densities
    ], dtype=np.float64)


def network_input(pairs, model):
    parameters = model.input_parameters
    normalized = parameters / model.scale
    p_input = np.broadcast_to(normalized, (len(pairs), len(parameters)))
    rho_p = pairs[:, 0:1].astype(np.float32)
    rho_n = pairs[:, 1:2].astype(np.float32)
    return np.concatenate([p_input, rho_n / 0.16, rho_p / 0.16], axis=1)


def benchmark_pinn(model, x, repeats):
    for _ in range(5):
        model(x)
    timings = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = model(x)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings)), output


def benchmark_exact(pairs, n_density):
    solutions = []
    failures = 0
    evaluations = 0
    start = time.perf_counter()
    for index, (rho_p, rho_n) in enumerate(pairs):
        if index % n_density == 0:
            guess = np.asarray([1.0, 1.0, 0.0, 0.0])
        solution, info, ier, _ = fsolve(
            residual, guess, args=(float(rho_p), float(rho_n)),
            xtol=1.0e-10, maxfev=500, full_output=True,
        )
        evaluations += int(info.get("nfev", 0))
        failures += int(ier != 1 or not np.all(np.isfinite(solution)))
        solutions.append(solution)
        guess = solution
    return time.perf_counter() - start, np.asarray(solutions), failures, evaluations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--density-points", type=int, default=100)
    parser.add_argument("--fraction-points", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=31)
    args = parser.parse_args()

    model = NumpyMLP(args.checkpoint)
    pairs = build_grid(args.density_points, args.fraction_points)
    x = network_input(pairs, model)
    pinn_time, prediction = benchmark_pinn(model, x, args.repeats)
    for _ in range(5): benchmark_exact(pairs,args.density_points)
    conventional=[]
    for _ in range(args.repeats):
        elapsed,exact_fields,failures,evaluations=benchmark_exact(pairs,args.density_points)
        if failures: raise RuntimeError(f'{failures} failed conventional solves; speedup not accepted')
        conventional.append(elapsed)
    exact_time=float(np.median(conventional))
    difference = np.abs(prediction - exact_fields)
    result = {
        "checkpoint_step": model.step,
        "conventional_repeat_seconds": conventional,
        "n_points": int(len(pairs)),
        "grid": {
            "density_points": args.density_points,
            "fraction_points": args.fraction_points,
            "rho_b_range_fm3": [0.005, 1.2],
            "proton_fraction_range": [0.02, 0.50],
        },
        "exact_continuation_single_cpu": {
            "wall_seconds": exact_time,
            "microseconds_per_point": 1.0e6 * exact_time / len(pairs),
            "function_evaluations": evaluations,
            "failed_points": failures,
        },
        "pinn_numpy_single_cpu_batch": {
            "median_wall_seconds": pinn_time,
            "microseconds_per_point": 1.0e6 * pinn_time / len(pairs),
            "speedup": exact_time / pinn_time,
        },
        "field_difference_for_timing_grid": {
            "mean_absolute_MeV": np.mean(difference, axis=0).tolist(),
            "max_absolute_MeV": np.max(difference, axis=0).tolist(),
            "field_order": ["sigma", "omega", "rho", "a0"],
        },
        "note": (
            "Both timings use one CPU thread. PINN timing is a prepared-batch forward "
            "pass reconstructed exactly from the PyTorch checkpoint. The exact baseline "
            "uses SciPy fsolve with density continuation at each proton fraction. EOS "
            "construction and TOV integration are excluded."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

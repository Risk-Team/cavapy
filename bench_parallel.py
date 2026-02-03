"""Lightweight benchmark for cavapy parallelization strategies."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass

import cavapy
from cava_config import VALID_GCM, VALID_RCM


@dataclass
class Scenario:
    name: str
    kwargs: dict


def _first_model_combo() -> tuple[str, str]:
    return VALID_GCM[0], VALID_RCM[0]


def _first_n_model_combos(n: int) -> list[tuple[str, str]]:
    combos = [(g, r) for g in VALID_GCM for r in VALID_RCM]
    return combos[:n]


def _run_once(name: str, **kwargs) -> float:
    start = time.perf_counter()
    cavapy.get_climate_data(**kwargs)
    return time.perf_counter() - start


def main() -> None:
    country = "Togo"
    cordex_domain = "AFR-22"
    years_up_to = 2015
    rcp = "rcp26"
    variables_small = ["rsds", "sfcWind", "hurs"]

    gcm_one, rcm_one = _first_model_combo()
    combos_three = _first_n_model_combos(3)
    gcm_list = [g for g, _ in combos_three]
    rcm_list = [r for _, r in combos_three]

    scenarios: list[Scenario] = [
        # Single model scenarios - test thread count impact
        Scenario(
            name="single_model_three_vars_threads_1",
            kwargs=dict(
                country=country,
                cordex_domain=cordex_domain,
                rcp=rcp,
                gcm=gcm_one,
                rcm=rcm_one,
                years_up_to=years_up_to,
                historical=True,
                variables=variables_small,
                max_threads_per_process=1,
            ),
        ),
        Scenario(
            name="single_model_three_vars_threads_3",
            kwargs=dict(
                country=country,
                cordex_domain=cordex_domain,
                rcp=rcp,
                gcm=gcm_one,
                rcm=rcm_one,
                years_up_to=years_up_to,
                historical=True,
                variables=variables_small,
                max_threads_per_process=3,
            ),
        ),
        # Multi-model scenarios - test process count impact
        Scenario(
            name="three_models_three_vars_processes_2_threads_2",
            kwargs=dict(
                country=country,
                cordex_domain=cordex_domain,
                rcp=rcp,
                gcm=gcm_list,
                rcm=rcm_list,
                years_up_to=years_up_to,
                historical=True,
                variables=variables_small,
                max_model_processes=2,
                max_threads_per_process=2,
            ),
        ),
        Scenario(
            name="three_models_three_vars_processes_4_threads_2",
            kwargs=dict(
                country=country,
                cordex_domain=cordex_domain,
                rcp=rcp,
                gcm=gcm_list,
                rcm=rcm_list,
                years_up_to=years_up_to,
                historical=True,
                variables=variables_small,
                max_model_processes=4,
                max_threads_per_process=2,
            ),
        ),
    ]

    results: list[dict[str, str]] = []
    for scenario in scenarios:
        print(f"Running: {scenario.name}")
        elapsed = _run_once(**scenario.kwargs, name=scenario.name)
        results.append({"scenario": scenario.name, "seconds": f"{elapsed:.2f}"})
        print(f"Done: {scenario.name} in {elapsed:.2f}s")

    output_path = "bench_parallel_results.csv"
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["scenario", "seconds"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()

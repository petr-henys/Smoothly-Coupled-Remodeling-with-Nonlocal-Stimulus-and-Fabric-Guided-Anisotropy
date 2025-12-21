"""Predefined gait loading scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from simulation.loader import HipLoadSpec, LoadingCase, MuscleLoadSpec


def get_standard_gait_cases() -> list[LoadingCase]:
    """Standard walking cycle: heel strike, mid-stance, toe-off, stair climb."""
    # 1) Heel strike (0–10%): impact/weight acceptance
    case_heel_strike = LoadingCase(
        name="heel_strike",
        day_cycles=1.0,
        hip=HipLoadSpec(
            magnitude=1900,
            alpha_sag=-10.0,
            alpha_front=-5.0,
            sigma_deg=25.0,
            flip=True,
        ),
        muscles=[
            MuscleLoadSpec(
                name="glmax",
                magnitude=900,
                alpha_sag=-50.0,
                alpha_front=20.0,
                sigma=5.0,
                flip=False,
            ),
            MuscleLoadSpec(
                name="vastus_lateralis",
                magnitude=800,
                alpha_sag=0.0,
                alpha_front=10.0,
                sigma=5.0,
                flip=True,
            ),
            MuscleLoadSpec(
                name="glmed",
                magnitude=500,
                alpha_sag=-10.0,
                alpha_front=30.0,
                sigma=4.0,
                flip=False,
            ),
        ],
    )

    # 2) Mid-stance (~30%): peak single-leg support load
    case_mid_stance = LoadingCase(
        name="mid_stance",
        day_cycles=1.0,
        hip=HipLoadSpec(
            magnitude=2400,
            alpha_sag=0.0,
            alpha_front=-5.0,
            sigma_deg=25.0,
            flip=True,
        ),
        muscles=[
            MuscleLoadSpec(
                name="glmed",
                magnitude=1300,
                alpha_sag=-5.0,
                alpha_front=15.0,
                sigma=3.0,
                flip=False,
            ),
            MuscleLoadSpec(
                name="glmin",
                magnitude=500,
                alpha_sag=10.0,
                alpha_front=10.0,
                sigma=3.0,
                flip=False,
            ),
            MuscleLoadSpec(
                name="vastus_lateralis",
                magnitude=200,
                alpha_sag=0.0,
                alpha_front=5.0,
                sigma=4.0,
                flip=True,
            ),
        ],
    )

    # 3) Toe-off (~60%): propulsion / pre-swing
    case_toe_off = LoadingCase(
        name="toe_off",
        day_cycles=1.0,
        hip=HipLoadSpec(
            magnitude=2100,
            alpha_sag=10.0,
            alpha_front=-5.0,
            sigma_deg=25.0,
            flip=True,
        ),
        muscles=[
            MuscleLoadSpec(
                name="psoas",
                magnitude=800,
                alpha_sag=35.0,
                alpha_front=15.0,
                sigma=3.0,
                flip=False,
            ),
            MuscleLoadSpec(
                name="vastus_lateralis",
                magnitude=300,
                alpha_sag=0.0,
                alpha_front=5.0,
                sigma=4.0,
                flip=True,
            ),
            MuscleLoadSpec(
                name="glmed",
                magnitude=300,
                alpha_sag=0.0,
                alpha_front=15.0,
                sigma=3.0,
                flip=False,
            ),
        ],
    )

    # 4) Stair climbing: high torsion/bending (low daily frequency)
    case_stair_climb = LoadingCase(
        name="stair_climb",
        day_cycles=0.1,
        hip=HipLoadSpec(
            magnitude=2500,
            alpha_sag=-30.0,
            alpha_front=30.0,
            sigma_deg=25.0,
            flip=True,
        ),
        muscles=[
            MuscleLoadSpec(
                name="glmax",
                magnitude=1200,
                alpha_sag=-60.0,
                alpha_front=20.0,
                sigma=5.0,
                flip=False,
            ),
            MuscleLoadSpec(
                name="vastus_lateralis",
                magnitude=1000,
                alpha_sag=0.0,
                alpha_front=10.0,
                sigma=5.0,
                flip=True,
            ),
        ],
    )

    return [case_heel_strike, case_mid_stance, case_toe_off, case_stair_climb]


def load_scenarios_from_yaml(path: str | Path) -> list[LoadingCase]:
    """Load custom loading scenarios from a YAML configuration file.

    Args:
        path: Path to YAML file with loading case definitions.

    Returns:
        List of LoadingCase objects.

    Example YAML format:
        ```yaml
        scenarios:
          - name: heel_strike
            day_cycles: 1.0
            hip:
              magnitude: 1900
              alpha_sag: -10.0
              alpha_front: -5.0
              sigma_deg: 25.0
              flip: true
            muscles:
              - name: glmax
                magnitude: 900
                alpha_sag: -50.0
                alpha_front: 20.0
                sigma: 5.0
                flip: false
        ```
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenarios file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return _parse_scenarios(data.get("scenarios", []))


def _parse_scenarios(scenarios_data: list[dict[str, Any]]) -> list[LoadingCase]:
    """Parse scenario dictionaries into LoadingCase objects."""
    cases = []
    for sc in scenarios_data:
        hip_data = sc.get("hip")
        hip = (
            HipLoadSpec(
                magnitude=hip_data["magnitude"],
                alpha_sag=hip_data["alpha_sag"],
                alpha_front=hip_data["alpha_front"],
                sigma_deg=hip_data["sigma_deg"],
                flip=hip_data.get("flip", True),
            )
            if hip_data
            else None
        )

        muscles = [
            MuscleLoadSpec(
                name=m["name"],
                magnitude=m["magnitude"],
                alpha_sag=m["alpha_sag"],
                alpha_front=m["alpha_front"],
                sigma=m["sigma"],
                flip=m.get("flip", False),
            )
            for m in sc.get("muscles", [])
        ]

        cases.append(
            LoadingCase(
                name=sc["name"],
                day_cycles=sc.get("day_cycles", 1.0),
                hip=hip,
                muscles=muscles,
            )
        )

    return cases

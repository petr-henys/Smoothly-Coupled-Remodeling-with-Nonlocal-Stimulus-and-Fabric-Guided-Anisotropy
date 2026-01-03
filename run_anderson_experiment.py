#!/usr/bin/env python3
"""Run the Anderson acceleration experiment comparing Physio vs Stiff configurations.

Runs 4 simulations:
    1. Physio + Anderson: Standard config, should converge easily
    2. Physio + Picard: May converge slowly, but should succeed
    3. Stiff + Anderson: Challenging config, AA helps it converge
    4. Stiff + Picard: Expected to FAIL (hit max_subiters)

The "stiff" config uses narrow lazy zone (δ₀=0.01), narrow saturation (κ=0.1),
and faster reaction rates - creating a reaction-dominated regime that breaks Picard.

Usage:
    mpirun -n 4 python run_anderson_experiment.py

Output directories:
    .physio_results_box/         - Physio + Anderson
    .physio_results_box_picard/  - Physio + Picard
    .stiff_results_box/          - Stiff + Anderson
    .stiff_results_box_picard/   - Stiff + Picard
"""

from __future__ import annotations

from pathlib import Path

from mpi4py import MPI

from box import (
    BoxSolverFactory,
    BoxLoader,
    BoxGeometry,
    BoxMeshBuilder,
    get_parabolic_pressure_case,
)
from simulation.model import Remodeller
from simulation.params import load_default_params, create_config
from simulation.progress import ProgressReporter
from simulation.storage import UnifiedStorage


EXPERIMENTS = [
    ("Physio + Anderson", "physio_params_box.json", ".physio_results_box", "anderson"),
    ("Physio + Picard", "physio_params_box.json", ".physio_results_box_picard", "picard"),
    ("Stiff + Anderson", "stiff_params_box.json", ".stiff_results_box", "anderson"),
    ("Stiff + Picard", "stiff_params_box.json", ".stiff_results_box_picard", "picard"),
]


def run_simulation(comm: MPI.Comm, params_file: str, output_dir: str, 
                   accel_type: str, run_name: str) -> bool:
    """Run a single box simulation."""
    rank = comm.rank
    
    if rank == 0:
        print(f"\n{'='*60}\n{run_name}\n  {params_file} → {output_dir} ({accel_type})\n{'='*60}")
    
    try:
        params = load_default_params(params_file)
        box = params["box"]
        
        params["output"].results_dir = output_dir
        params["solver"].accel_type = accel_type
        params["geometry"].fix_tag = BoxMeshBuilder.TAG_BOTTOM
        params["geometry"].load_tag = BoxMeshBuilder.TAG_TOP
        
        geometry = BoxGeometry(Lx=box["Lx"], Ly=box["Ly"], Lz=box["Lz"],
                               nx=box["nx"], ny=box["ny"], nz=box["nz"])
        domain, facet_tags = BoxMeshBuilder(geometry, comm).build()
        cfg = create_config(domain, facet_tags, params)
        
        if rank == 0:
            Path(cfg.log_file).parent.mkdir(parents=True, exist_ok=True)
            Path(cfg.log_file).write_text(f"# {run_name}\n")
        
        loading_cases = [get_parabolic_pressure_case(
            pressure=box["pressure"], load_tag=BoxMeshBuilder.TAG_TOP,
            gradient_axis=box["gradient_axis"], center_factor=box["center_factor"],
            edge_factor=box["edge_factor"], box_extent=(0.0, box["Lx"]),
            name="parabolic_compression",
        )]
        
        loader = BoxLoader(domain, facet_tags, loading_cases=loading_cases)
        loader.set_loading_case(loading_cases[0].name)
        
        storage = UnifiedStorage(cfg)
        storage.fields.register("traction", [loader.traction], filename="traction.bp")
        storage.fields.write("traction", 0.0)
        storage.close()
        
        with Remodeller(cfg, loader=loader, factory=BoxSolverFactory(cfg)) as model:
            with ProgressReporter(comm, cfg.time.total_time, cfg.solver.max_subiters) as reporter:
                model.simulate(reporter=reporter)
        
        if rank == 0:
            print(f"✓ {run_name} completed")
        return True
        
    except Exception as e:
        if rank == 0:
            print(f"✗ {run_name} failed: {e}")
        return False


def main() -> None:
    comm = MPI.COMM_WORLD
    
    if comm.rank == 0:
        print("=" * 70)
        print("Anderson Acceleration Experiment")
        print("=" * 70)
    
    results = []
    for name, params, output, accel in EXPERIMENTS:
        success = run_simulation(comm, params, output, accel, name)
        results.append((name, success))
        comm.Barrier()
    
    if comm.rank == 0:
        print("\n" + "=" * 70)
        print("Summary:")
        for name, success in results:
            print(f"  {name}: {'✓' if success else '✗'}")
        print("\nNext: python analysis/anderson_comparison_plot.py")
        print("=" * 70)


if __name__ == "__main__":
    main()

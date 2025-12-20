"""Parameter sweep framework with hash-based output naming."""

import csv
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Protocol
from mpi4py import MPI
from tqdm import tqdm

from simulation.logger import get_logger

ParamValue = int | float | str | bool | None
ParamDict = Dict[str, List[ParamValue]]


class Runnable(Protocol):
    def __call__(self, param_point: Dict[str, ParamValue], output_path: Path, comm: MPI.Comm) -> None: ...


@dataclass
class ParameterSweep:
    """Parameter sweep with Cartesian product and hash-based paths."""
    params: ParamDict
    base_output_dir: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.params:
            raise ValueError("params dictionary cannot be empty")
        
        for key, values in self.params.items():
            if not isinstance(values, list):
                self.params[key] = [values]
        
        self.base_output_dir = Path(self.base_output_dir)
    
    def generate_points(self) -> List[Dict[str, ParamValue]]:
        """Cartesian product of all parameter values."""
        param_names = list(self.params.keys())
        param_values = [self.params[k] for k in param_names]
        return [dict(zip(param_names, values)) for values in itertools.product(*param_values)]
    
    def format_output_path(self, param_point: Dict[str, ParamValue]) -> Path:
        """Output path via 8-char hash of parameter point."""
        param_str = json.dumps(param_point, sort_keys=True, separators=(",", ":"))
        params_hash = hashlib.sha1(param_str.encode("utf-8")).hexdigest()[:8]
        return self.base_output_dir / params_hash
    
    def total_runs(self) -> int:
        """Calculate total number of runs in sweep."""
        total = 1
        for values in self.params.values():
            total *= len(values)
        return total

class Parametrizer:
    """Execute parameter sweeps with arbitrary callables."""
    
    def __init__(
        self,
        sweep: ParameterSweep,
        callable_func: Runnable,
        comm: MPI.Comm,
        verbose: bool = True
    ):
        self.sweep = sweep
        self.callable_func = callable_func
        self.comm = comm
        self.logger = get_logger(comm, name="Parametrizer")
    
    def run(self) -> None:
        """Execute parameter sweep and save summary."""
        param_points = self.sweep.generate_points()
        total_runs = len(param_points)
        
        if self.comm.rank == 0:
            self.logger.info(f"Parameter sweep: {total_runs} runs")
        
        runs_data: List[Dict[str, Any]] = []
        
        iterator = enumerate(param_points)
        if self.comm.rank == 0:
            iterator = tqdm(iterator, total=total_runs, desc="Parameter Sweep", unit="run", ncols=100)
        
        for idx, param_point in iterator:
            output_path = self.sweep.format_output_path(param_point)
            self.callable_func(param_point, output_path, self.comm)
            
            runs_data.append({
                "output_dir": output_path.name,
                **param_point
            })
        
        self._save_summary(runs_data)
        
        if self.comm.rank == 0:
            self.logger.info(f"Sweep complete. Summary: {self.sweep.base_output_dir / 'sweep_summary.csv'}")
    
    def _save_summary(self, runs_data: List[Dict[str, Any]]) -> None:
        """Save parameter sweep summary (rank 0 only)."""
        if self.comm.rank != 0:
            return
        
        if not runs_data:
            return
        
        self.sweep.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = self.sweep.base_output_dir / "sweep_summary.csv"
        json_file = self.sweep.base_output_dir / "sweep_summary.json"
        
        # Write CSV summary.
        header = list(runs_data[0].keys())
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(runs_data)
        
        # Write JSON summary.
        summary = {
            "metadata": self.sweep.metadata,
            "total_runs": len(runs_data),
            "runs": runs_data
        }
        with open(json_file, "w") as f:
            json.dump(summary, f, indent=2)
    
    def run_single(self, param_point: Dict[str, ParamValue], output_path: Path | None = None) -> None:
        """Execute single run with given parameters."""
        if output_path is None:
            output_path = self.sweep.format_output_path(param_point)
        self.callable_func(param_point, output_path, self.comm)

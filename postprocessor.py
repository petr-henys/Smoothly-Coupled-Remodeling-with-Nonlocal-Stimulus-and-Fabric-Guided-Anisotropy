from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from mpi4py import MPI
from dolfinx import fem

from analysis.utils import load_npz_field
from simulation.logger import get_logger


class SimulationLoader:
    """Load and analyze results from a parametrizer sweep run directory.
    
    Provides MPI-parallel access to configuration, fields, and metrics.
    
    Directory structure expected:
        output_dir/
            config.json          # Configuration parameters
            run_summary.json     # Run metadata
            steps.csv            # Per-timestep metrics
            subiterations.csv    # Per-subiteration metrics
            u.npz, rho.npz, S.npz, A.npz  # Field snapshots
            u.bp/, scalars.bp/, A.bp/     # VTX field time series (optional)
    """
    
    __slots__ = (
        "output_dir", "comm", "logger", "_config", "_run_summary",
        "_steps_df", "_subiters_df", "_field_times", "_field_cache"
    )
    
    def __init__(self, output_dir: str | Path, comm: MPI.Comm, verbose: bool = True):
        """Initialize loader for a single simulation run directory.
        
        Args:
            output_dir: Path to run directory (hash-named subdirectory from parametrizer)
            comm: MPI communicator
            verbose: Enable logging
        """
        self.output_dir = Path(output_dir)
        self.comm = comm
        self.logger = get_logger(comm, verbose=verbose, name="SimulationLoader")
        
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Simulation directory not found: {self.output_dir}")
        
        # Lazy-loaded caches
        self._config: Optional[Dict[str, Any]] = None
        self._run_summary: Optional[Dict[str, Any]] = None
        self._steps_df: Optional[pd.DataFrame] = None
        self._subiters_df: Optional[pd.DataFrame] = None
        self._field_times: Optional[Dict[str, np.ndarray]] = None
        self._field_cache: Dict[Tuple[str, float], np.ndarray] = {}
        
        self.logger.debug(lambda: f"Initialized loader for {self.output_dir}")
    
    # ========================================================================
    # Configuration and metadata
    # ========================================================================
    
    def get_config(self) -> Dict[str, Any]:
        """Load full configuration dictionary from config.json.
        
        Returns:
            Dictionary with all configuration parameters
        """
        if self._config is None:
            config_file = self.output_dir / "config.json"
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            
            if self.comm.rank == 0:
                with open(config_file, "r") as f:
                    self._config = json.load(f)
            else:
                self._config = None
            
            self._config = self.comm.bcast(self._config, root=0)
        
        return self._config
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get single configuration parameter value.
        
        Args:
            key: Parameter name
            default: Default value if key not found
            
        Returns:
            Parameter value or default
        """
        config = self.get_config()
        return config.get(key, default)
    
    def get_run_summary(self) -> Dict[str, Any]:
        """Load run summary metadata from run_summary.json.
        
        Contains runtime statistics like total time, convergence info, etc.
        
        Returns:
            Dictionary with run summary metadata
        """
        if self._run_summary is None:
            summary_file = self.output_dir / "run_summary.json"
            if not summary_file.exists():
                self.logger.warning(f"Run summary not found: {summary_file}")
                return {}
            
            if self.comm.rank == 0:
                with open(summary_file, "r") as f:
                    self._run_summary = json.load(f)
            else:
                self._run_summary = None
            
            self._run_summary = self.comm.bcast(self._run_summary, root=0)
        
        return self._run_summary
    
    # ========================================================================
    # Metrics (CSV data)
    # ========================================================================
    
    def get_steps_metrics(self) -> pd.DataFrame:
        """Load per-timestep metrics from steps.csv.
        
        Columns include: step, time_days, dt_days, solver iterations, memory, etc.
        
        Returns:
            DataFrame with one row per timestep
        """
        if self._steps_df is None:
            steps_file = self.output_dir / "steps.csv"
            if not steps_file.exists():
                raise FileNotFoundError(f"Steps metrics not found: {steps_file}")
            
            if self.comm.rank == 0:
                self._steps_df = pd.read_csv(steps_file)
            else:
                self._steps_df = None
            
            # Broadcast (small dataframe, OK to replicate)
            self._steps_df = self.comm.bcast(self._steps_df, root=0)
        
        return self._steps_df
    
    def get_subiterations_metrics(self) -> pd.DataFrame:
        """Load per-subiteration metrics from subiterations.csv.
        
        Columns include: step, iter, time_days, residuals, convergence info,
        subsolver timings, KSP iterations, conservation checks, memory, etc.
        
        Returns:
            DataFrame with one row per subiteration
        """
        if self._subiters_df is None:
            subiters_file = self.output_dir / "subiterations.csv"
            if not subiters_file.exists():
                raise FileNotFoundError(f"Subiterations metrics not found: {subiters_file}")
            
            if self.comm.rank == 0:
                self._subiters_df = pd.read_csv(subiters_file)
            else:
                self._subiters_df = None
            
            # Broadcast (potentially large, but needed for analysis)
            self._subiters_df = self.comm.bcast(self._subiters_df, root=0)
        
        return self._subiters_df
    
    def get_metrics_at_time(self, time_days: float) -> Dict[str, Any]:
        """Get all metrics for the timestep closest to requested time.
        
        Args:
            time_days: Target time in days
            
        Returns:
            Dictionary with metrics from steps.csv and aggregated subiterations
        """
        steps_df = self.get_steps_metrics()
        subiters_df = self.get_subiterations_metrics()
        
        # Find closest timestep
        time_col = "time_days"
        if time_col not in steps_df.columns:
            raise ValueError(f"Column '{time_col}' not found in steps.csv")
        
        idx = (steps_df[time_col] - time_days).abs().idxmin()
        step_metrics = steps_df.loc[idx].to_dict()
        
        # Get corresponding subiterations
        step_num = int(step_metrics["step"])
        step_subiters = subiters_df[subiters_df["step"] == step_num]
        
        # Aggregate subiteration stats
        subiters_summary = {
            "num_subiters": len(step_subiters),
            "final_proj_res": step_subiters["proj_res"].iloc[-1] if len(step_subiters) > 0 else np.nan,
            "total_mech_iters": step_subiters["mech_iters"].sum() if "mech_iters" in step_subiters else 0,
            "total_stim_iters": step_subiters["stim_iters"].sum() if "stim_iters" in step_subiters else 0,
            "total_dens_iters": step_subiters["dens_iters"].sum() if "dens_iters" in step_subiters else 0,
            "total_dir_iters": step_subiters["dir_iters"].sum() if "dir_iters" in step_subiters else 0,
            "aa_acceptances": step_subiters["accepted"].sum() if "accepted" in step_subiters else 0,
            "aa_restarts": step_subiters["restart"].sum() if "restart" in step_subiters else 0,
        }
        
        return {**step_metrics, **subiters_summary}
    
    # ========================================================================
    # Field loading (NPZ snapshots)
    # ========================================================================
    
    def _get_field_times(self) -> Dict[str, np.ndarray]:
        """Extract available time checkpoints from steps.csv.
        
        Returns:
            Dictionary mapping field names to arrays of available times
        """
        if self._field_times is None:
            steps_df = self.get_steps_metrics()
            times = steps_df["time_days"].values
            
            # All four fields saved together at same checkpoints
            self._field_times = {
                "u": times.copy(),
                "rho": times.copy(),
                "S": times.copy(),
                "A": times.copy(),
            }
        
        return self._field_times
    
    def get_available_times(self) -> np.ndarray:
        """Get array of all available field checkpoint times.
        
        Returns:
            Sorted array of time values in days
        """
        times_dict = self._get_field_times()
        return times_dict["u"]  # All fields have same times
    
    def _load_field_raw(self, field_name: str, time_days: float) -> np.ndarray:
        """Load field snapshot from NPZ file at exact checkpoint time.
        
        Args:
            field_name: Field name ('u', 'rho', 'S', 'A')
            time_days: Time checkpoint (must exist exactly)
            
        Returns:
            DOF values array (1D, including ghost DOFs)
        """
        npz_file = self.output_dir / f"{field_name}.npz"
        if not npz_file.exists():
            raise FileNotFoundError(f"Field snapshot not found: {npz_file}")
        
        # NPZ files contain all checkpoints - need to implement time selection
        # For now, assume NPZ contains only final state (as in run_anderson.py)
        # Full time series loading would require VTX reader or extended NPZ format
        
        if self.comm.rank == 0:
            with np.load(npz_file) as data:
                values = data["values"].copy()
        else:
            values = None
        
        values = self.comm.bcast(values, root=0)
        return values
    
    def get_fields_at_time(
        self,
        time_days: float,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Load all fields at requested time.
        
        Args:
            time_days: Target time in days (must match a checkpoint)
            fields: List of field names to load (default: all four fields)
            
        Returns:
            Dictionary mapping field names to DOF value arrays
            
        Raises:
            ValueError: If requested time does not match any checkpoint
            
        Note:
            Current implementation loads final state from NPZ files.
        """
        if fields is None:
            fields = ["u", "rho", "S", "A"]
        
        available_times = self.get_available_times()
        
        # Check if exact time exists
        if not np.any(np.isclose(available_times, time_days, rtol=1e-9)):
            raise ValueError(
                f"Time {time_days} days not found in checkpoints. "
                f"Available times: {available_times}"
            )
        
        # Load exact checkpoint
        exact_time = available_times[np.argmin(np.abs(available_times - time_days))]
        result = {}
        for field_name in fields:
            cache_key = (field_name, exact_time)
            if cache_key not in self._field_cache:
                self._field_cache[cache_key] = self._load_field_raw(field_name, exact_time)
            result[field_name] = self._field_cache[cache_key]
        
        return result
    
    def load_field_to_function(
        self,
        target: fem.Function,
        field_name: str,
        time_days: Optional[float] = None,
    ) -> None:
        """Load field snapshot into a DOLFINx function.
        
        Uses coordinate-based DOF matching from analysis.utils.load_npz_field,
        making loading MPI-independent (works with any mesh partition).
        
        Args:
            target: DOLFINx Function to populate (must have compatible element)
            field_name: Field name ('u', 'rho', 'S', 'A')
            time_days: Target time (default: use final checkpoint)
            
        Raises:
            RuntimeError: If element type mismatch between stored and target
        """
        if time_days is None:
            # Use final checkpoint
            available_times = self.get_available_times()
            time_days = available_times[-1]
        
        npz_file = self.output_dir / f"{field_name}.npz"
        if not npz_file.exists():
            raise FileNotFoundError(f"Field snapshot not found: {npz_file}")
        
        # Use analysis.utils coordinate-matching loader
        load_npz_field(self.comm, npz_file, target)
        
        self.logger.debug(
            lambda: f"Loaded {field_name} at t={time_days:.1f} days into function"
        )
    
    # ========================================================================
    # Convenience methods
    # ========================================================================
    
    def get_field_statistics(
        self,
        field_name: str,
        time_days: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute basic statistics for a field at given time.
        
        Args:
            field_name: Field name ('u', 'rho', 'S', 'A')
            time_days: Target time (default: final checkpoint)
            
        Returns:
            Dictionary with min, max, mean, std statistics
        """
        fields = self.get_fields_at_time(
            time_days if time_days is not None else self.get_available_times()[-1],
            fields=[field_name]
        )
        values = fields[field_name]
        
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
        }


# ============================================================================
# Sweep-level utilities
# ============================================================================

class SweepLoader:
    """Load and analyze multiple runs from a parametrizer sweep.
    
    Provides access to all runs in a sweep directory with filtering by
    parameter values.
    
    Usage:
        sweep = SweepLoader("results/anderson_sweep", comm)
        
        # Get all runs with specific parameters
        picard_runs = sweep.filter_runs(accel_type="picard")
        
        # Load specific run
        loader = sweep.get_loader("111159cb")
    """
    
    __slots__ = ("base_dir", "comm", "logger", "_summary_df", "_loaders")
    
    def __init__(self, base_dir: str | Path, comm: MPI.Comm, verbose: bool = True):
        """Initialize sweep loader.
        
        Args:
            base_dir: Path to sweep base directory (contains hash subdirectories)
            comm: MPI communicator
            verbose: Enable logging
        """
        self.base_dir = Path(base_dir)
        self.comm = comm
        self.logger = get_logger(comm, verbose=verbose, name="SweepLoader")
        
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Sweep directory not found: {self.base_dir}")
        
        self._summary_df: Optional[pd.DataFrame] = None
        self._loaders: Dict[str, SimulationLoader] = {}
    
    def get_summary(self) -> pd.DataFrame:
        """Load sweep summary with all parameter combinations.
        
        Returns:
            DataFrame with columns: output_dir, and all swept parameters
        """
        if self._summary_df is None:
            summary_file = self.base_dir / "sweep_summary.csv"
            
            if summary_file.exists():
                if self.comm.rank == 0:
                    self._summary_df = pd.read_csv(summary_file)
                else:
                    self._summary_df = None
                self._summary_df = self.comm.bcast(self._summary_df, root=0)
            else:
                # Fallback: scan directory for hash subdirs
                self.logger.warning("sweep_summary.csv not found, scanning directories")
                
                if self.comm.rank == 0:
                    subdirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
                    self._summary_df = pd.DataFrame([{"output_dir": d.name} for d in subdirs])
                else:
                    self._summary_df = None
                self._summary_df = self.comm.bcast(self._summary_df, root=0)
        
        return self._summary_df
    
    def filter_runs(self, **param_filters) -> pd.DataFrame:
        """Filter runs by parameter values.
        
        Args:
            **param_filters: Parameter name-value pairs to filter by
            
        Returns:
            Filtered DataFrame subset
            
        Example:
            sweep.filter_runs(accel_type="anderson", dt_days=50.0)
        """
        summary = self.get_summary()
        mask = pd.Series([True] * len(summary))
        
        for param, value in param_filters.items():
            if param in summary.columns:
                mask &= (summary[param] == value)
            else:
                self.logger.warning(f"Parameter '{param}' not found in sweep summary")
        
        return summary[mask]
    
    def get_loader(self, run_hash: str) -> SimulationLoader:
        """Get loader for specific run by hash.
        
        Args:
            run_hash: Run directory hash (8-character)
            
        Returns:
            SimulationLoader instance
        """
        if run_hash not in self._loaders:
            run_dir = self.base_dir / run_hash
            self._loaders[run_hash] = SimulationLoader(run_dir, self.comm, verbose=False)
        
        return self._loaders[run_hash]
    
    def get_all_loaders(self) -> List[SimulationLoader]:
        """Get loaders for all runs in sweep.
        
        Returns:
            List of SimulationLoader instances
        """
        summary = self.get_summary()
        return [self.get_loader(row["output_dir"]) for _, row in summary.iterrows()]

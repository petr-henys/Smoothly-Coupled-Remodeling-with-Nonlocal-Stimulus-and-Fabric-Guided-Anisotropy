#!/usr/bin/env python3
"""Properly merge test files by concatenating them."""
from pathlib import Path

def merge_files(sources, target, header_comment):
    """Merge source files into target with proper header."""
    content_parts = []
    
    # Read all source files
    for src in sources:
        src_path = Path(src)
        if not src_path.exists():
            print(f"  WARNING: {src} not found")
            continue
        content_parts.append(src_path.read_text())
    
    # Write merged file
    with open(target, 'w') as f:
        f.write(f'#!/usr/bin/env python3\n')
        f.write(f'"""\n{header_comment}\n\n')
        f.write('Merged from:\n')
        for src in sources:
            f.write(f'- {src}\n')
        f.write('"""\n\n')
        
        # Join with separators
        separator = '\n\n' + '#' * 80 + '\n\n'
        f.write(separator.join(content_parts))

# Define merges
merges = [
    (['test_numerical_implementation.py', 'test_nd_matrix_sanity.py'], 
     'test_numerical.py',
     'Numerical implementation and matrix assembly tests.'),
    
    (['test_nondimensional_params.py', 'test_nondimensional_subsolvers.py'],
     'test_nondimensionalization.py',
     'Nondimensionalization parameter consistency and subsolver scaling tests.'),
    
    (['test_fixedpoint_stop_and_anderson.py', 'test_coupling_tolerance.py'],
     'test_coupling.py',
     'Fixed-point coupling and Anderson acceleration tests.'),
    
    (['test_mpi_parallelism.py', 'test_performance.py'],
     'test_mpi_parallel.py',
     'MPI parallelism and performance tests.'),
    
    (['test_storage.py', 'test_logger.py', 'test_logging_monitoring.py'],
     'test_io_storage.py',
     'Storage, logging, and telemetry tests.'),
    
    (['test_convergence_analysis.py', 'test_convergence_npz_io.py', 
      'test_convergence_qoi_energy.py', 'test_npz_cross_mpi.py'],
     'test_convergence.py',
     'Convergence analysis, NPZ I/O, and QoI tests.'),
]

for sources, target, comment in merges:
    print(f'Creating {target}...')
    merge_files(sources, target, comment)
    print(f'  ✓ Created')

print('\nDone!')

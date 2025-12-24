#!/usr/bin/env python3
"""
Analyze runtime data to compute speedup between default ordering and PATCH_ORDERING.
Displays mesh characteristics (G_N, G_NNZ) alongside speedup for each mesh.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# Representative meshes mapping (short name -> full name in data)
REPRESENTATIVE_MESHES = {
    'fish': 'fish',
    'skull': 'skull',
    'nefertiti': 'nefertiti',
    'wingedvictory': 'wingedvictory',
    'blue-crab': 'blue-crab-master-geometry'
}


def analyze_representative_meshes(runtime_df, mesh_char_df, output_dir):
    """
    Analyze 5 representative meshes and create:
    1. A table showing speedup for these meshes
    2. Bar plot for ordering + analysis speedup
    3. Bar plot for factorization + solve speedup
    """
    # Get mesh characteristics (first DEFAULT row per mesh)
    mesh_char_default = mesh_char_df[mesh_char_df['ordering_type'] == 'DEFAULT']
    mesh_info = mesh_char_default.groupby('mesh_name').first()[['G_N', 'G_NNZ']]

    results = []

    for short_name, full_name in REPRESENTATIVE_MESHES.items():
        # Get default row (first occurrence)
        default_rows = runtime_df[
            (runtime_df['mesh_name'] == full_name) &
            (runtime_df['ordering_type'] == 'default')
        ]
        if default_rows.empty:
            print(f"Warning: No default data for {full_name}")
            continue
        default_row = default_rows.iloc[0]

        # Get best PATCH_ORDERING row (min total_time)
        patch_rows = runtime_df[
            (runtime_df['mesh_name'] == full_name) &
            (runtime_df['ordering_type'] == 'PATCH_ORDERING')
        ]
        if patch_rows.empty:
            print(f"Warning: No PATCH_ORDERING data for {full_name}")
            continue
        best_patch_idx = patch_rows['total_time'].idxmin()
        patch_row = patch_rows.loc[best_patch_idx]

        # Get mesh characteristics
        if full_name in mesh_info.index:
            g_n = mesh_info.loc[full_name, 'G_N']
            g_nnz = mesh_info.loc[full_name, 'G_NNZ']
        else:
            g_n = 0
            g_nnz = 0

        # Calculate speedups
        default_ordering_analysis = default_row['ordering_time'] + default_row['analysis_time']
        patch_ordering_analysis = patch_row['ordering_time'] + patch_row['analysis_time']
        ordering_analysis_speedup = default_ordering_analysis / patch_ordering_analysis

        default_factor_solve = default_row['factorization_time'] + default_row['solve_time']
        patch_factor_solve = patch_row['factorization_time'] + patch_row['solve_time']
        factor_solve_speedup = default_factor_solve / patch_factor_solve

        total_speedup = default_row['total_time'] / patch_row['total_time']

        results.append({
            'short_name': short_name,
            'full_name': full_name,
            'g_n': g_n,
            'g_nnz': g_nnz,
            'default_time': default_row['total_time'],
            'patch_time': patch_row['total_time'],
            'total_speedup': total_speedup,
            'ordering_analysis_speedup': ordering_analysis_speedup,
            'factor_solve_speedup': factor_solve_speedup,
            'default_ordering_analysis': default_ordering_analysis,
            'patch_ordering_analysis': patch_ordering_analysis,
            'default_factor_solve': default_factor_solve,
            'patch_factor_solve': patch_factor_solve
        })

    # Sort by G_N
    results.sort(key=lambda x: x['g_n'])

    # Print table
    print("\n" + "=" * 125)
    print("REPRESENTATIVE MESHES ANALYSIS")
    print("=" * 125)
    print(f"{'Mesh Name':<55} {'G_N':>12} {'G_NNZ':>14} {'Default (ms)':>14} {'Patch (ms)':>14} {'Speedup':>10}")
    print("-" * 125)

    for r in results:
        g_n_str = f"{r['g_n']:,}" if r['g_n'] != 0 else "N/A"
        g_nnz_str = f"{r['g_nnz']:,}" if r['g_nnz'] != 0 else "N/A"
        print(f"{r['short_name']:<55} {g_n_str:>12} {g_nnz_str:>14} {r['default_time']:>14.2f} {r['patch_time']:>14.2f} {r['total_speedup']:>10.2f}x")

    print("-" * 125)

    # Create bar plots
    short_names = [r['short_name'] for r in results]
    ordering_analysis_speedups = [r['ordering_analysis_speedup'] for r in results]
    factor_solve_speedups = [r['factor_solve_speedup'] for r in results]

    x = np.arange(len(short_names))
    width = 0.6

    # Plot 1: Ordering + Analysis Speedup
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars1 = ax1.bar(x, ordering_analysis_speedups, width, color='steelblue')
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=45, ha='right')
    ax1.axhline(y=1, color='red', linestyle='--', linewidth=1, label='No speedup (1x)')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend()

    # Add value labels on bars
    for bar, val in zip(bars1, ordering_analysis_speedups):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.2f}x', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    fig1.savefig(output_dir / 'ordering_analysis_speedup.pdf', dpi=150)
    print(f"\nSaved: {output_dir / 'ordering_analysis_speedup.pdf'}")

    # Plot 2: Factorization + Solve Speedup
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars2 = ax2.bar(x, factor_solve_speedups, width, color='darkorange')
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=45, ha='right')
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=1, label='No speedup (1x)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend()

    # Add value labels on bars
    for bar, val in zip(bars2, factor_solve_speedups):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.2f}x', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    fig2.savefig(output_dir / 'factorization_solve_speedup.pdf', dpi=150)
    print(f"Saved: {output_dir / 'factorization_solve_speedup.pdf'}")

    plt.close('all')

    return results


def create_iteration_speedup_table(runtime_df, mesh_char_df):
    """
    Create a table showing speedup at 1, 5, and 10 iterations,
    plus the maximum iteration count where PATCH_ORDERING is faster.
    
    Speedup(N) = (Default_Ordering + Default_Analysis + N × (Default_Factor + Default_Solve))
               / (Patch_Ordering + Patch_Analysis + N × (Patch_Factor + Patch_Solve))
    """
    # Get mesh characteristics (first DEFAULT row per mesh)
    mesh_char_default = mesh_char_df[mesh_char_df['ordering_type'] == 'DEFAULT']
    mesh_info = mesh_char_default.groupby('mesh_name').first()[['G_N', 'G_NNZ']]

    results = []

    for short_name, full_name in REPRESENTATIVE_MESHES.items():
        # Get default row (first occurrence)
        default_rows = runtime_df[
            (runtime_df['mesh_name'] == full_name) &
            (runtime_df['ordering_type'] == 'default')
        ]
        if default_rows.empty:
            continue
        default_row = default_rows.iloc[0]

        # Get best PATCH_ORDERING row (min total_time)
        patch_rows = runtime_df[
            (runtime_df['mesh_name'] == full_name) &
            (runtime_df['ordering_type'] == 'PATCH_ORDERING')
        ]
        if patch_rows.empty:
            continue
        best_patch_idx = patch_rows['total_time'].idxmin()
        patch_row = patch_rows.loc[best_patch_idx]

        # Get mesh characteristics for sorting
        g_n = mesh_info.loc[full_name, 'G_N'] if full_name in mesh_info.index else 0

        # Extract timing components
        default_setup = default_row['ordering_time'] + default_row['analysis_time']
        default_iter = default_row['factorization_time'] + default_row['solve_time']
        
        patch_setup = patch_row['ordering_time'] + patch_row['analysis_time']
        patch_iter = patch_row['factorization_time'] + patch_row['solve_time']

        # Compute speedup for N iterations
        def compute_speedup(n_iter):
            default_total = default_setup + n_iter * default_iter
            patch_total = patch_setup + n_iter * patch_iter
            return default_total / patch_total

        speedup_1 = compute_speedup(1)
        speedup_5 = compute_speedup(5)
        speedup_10 = compute_speedup(10)

        # Compute max iterations where Speedup > 1 (patch is faster)
        # Speedup > 1 when: default_setup + N*default_iter > patch_setup + N*patch_iter
        # Rearranging: (default_setup - patch_setup) > N * (patch_iter - default_iter)
        # Let: setup_diff = default_setup - patch_setup
        #      iter_diff = patch_iter - default_iter
        setup_diff = default_setup - patch_setup
        iter_diff = patch_iter - default_iter

        if iter_diff <= 0:
            # Patch is same or faster per iteration, so always faster (if setup is also faster)
            # or becomes faster eventually and stays faster
            max_iter = float('inf')
        elif setup_diff <= 0:
            # Patch setup is slower AND patch per-iter is slower -> never faster
            max_iter = 0
        else:
            # Patch setup is faster but per-iter is slower
            # Speedup > 1 when N < setup_diff / iter_diff
            max_iter = int(setup_diff / iter_diff)

        results.append({
            'short_name': short_name,
            'g_n': g_n,
            'speedup_1': speedup_1,
            'speedup_5': speedup_5,
            'speedup_10': speedup_10,
            'max_iter': max_iter
        })

    # Sort by G_N
    results.sort(key=lambda x: x['g_n'])

    # Print table
    print("\n" + "=" * 85)
    print("ITERATION SPEEDUP ANALYSIS")
    print("=" * 85)
    print(f"{'Mesh Name':<20} {'1 iter':>12} {'5 iter':>12} {'10 iter':>12} {'Max #iter':>15}")
    print("-" * 85)

    for r in results:
        max_iter_str = "inf" if r['max_iter'] == float('inf') else str(r['max_iter'])
        print(f"{r['short_name']:<20} {r['speedup_1']:>11.2f}x {r['speedup_5']:>11.2f}x {r['speedup_10']:>11.2f}x {max_iter_str:>15}")

    print("-" * 85)

    return results


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"

    # Load CSV files
    runtime_df = pd.read_csv(data_dir / "runtime_data.csv")
    mesh_char_df = pd.read_csv(data_dir / "mesh_char_data.csv")

    # Get default time (first occurrence per mesh)
    default_df = runtime_df[runtime_df['ordering_type'] == 'default']
    default_times = default_df.groupby('mesh_name')['total_time'].first()

    # Get min PATCH_ORDERING time per mesh
    patch_df = runtime_df[runtime_df['ordering_type'] == 'PATCH_ORDERING']
    patch_min_times = patch_df.groupby('mesh_name')['total_time'].min()

    # Get mesh characteristics (first DEFAULT row per mesh)
    mesh_char_default = mesh_char_df[mesh_char_df['ordering_type'] == 'DEFAULT']
    mesh_info = mesh_char_default.groupby('mesh_name').first()[['G_N', 'G_NNZ']]

    # Find common meshes that have both default and PATCH_ORDERING data
    common_meshes = default_times.index.intersection(patch_min_times.index)

    # Build results list for sorting by G_N
    results = []
    for mesh_name in common_meshes:
        default_time = default_times[mesh_name]
        patch_time = patch_min_times[mesh_name]
        speedup = default_time / patch_time

        # Get mesh characteristics if available
        if mesh_name in mesh_info.index:
            g_n = mesh_info.loc[mesh_name, 'G_N']
            g_nnz = mesh_info.loc[mesh_name, 'G_NNZ']
        else:
            g_n = 0  # Use 0 for sorting if not available
            g_nnz = 0

        results.append({
            'mesh_name': mesh_name,
            'g_n': g_n,
            'g_nnz': g_nnz,
            'default_time': default_time,
            'patch_time': patch_time,
            'speedup': speedup
        })

    # Sort by G_N
    results.sort(key=lambda x: x['g_n'])

    # Print header
    print(f"{'Mesh Name':<55} {'G_N':>12} {'G_NNZ':>14} {'Default (ms)':>14} {'Patch (ms)':>14} {'Speedup':>10}")
    print("-" * 125)

    # Process each mesh
    for r in results:
        g_n_str = r['g_n'] if r['g_n'] != 0 else "N/A"
        g_nnz_str = r['g_nnz'] if r['g_nnz'] != 0 else "N/A"
        print(f"{r['mesh_name']:<55} {g_n_str:>12} {g_nnz_str:>14} {r['default_time']:>14.2f} {r['patch_time']:>14.2f} {r['speedup']:>10.2f}x")

    print("-" * 125)
    print(f"\nTotal meshes analyzed: {len(results)}")

    # Analyze representative meshes and create bar plots
    analyze_representative_meshes(runtime_df, mesh_char_df, script_dir)

    # Create iteration speedup table
    create_iteration_speedup_table(runtime_df, mesh_char_df)


if __name__ == "__main__":
    main()

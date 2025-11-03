# Fat-Tree Topology Analysis

This project implements a three-tier Fat-Tree network topology (core, aggregation, edge) using NetworkX and provides several visualizations and analyses.

## Quick Start (Windows + VS Code)

1) Create and activate a virtual environment
```powershell
cd "c:\your_repository_folder\fat-tree-topology-analysis"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

2) Install libraries via requirements file
```powershell
pip install -r requirements.txt
```

3) Run fat_tree.py
```powershell
python .\fat_tree.py
```

4) Use the terminal for inputs
- Enter an even k (e.g., 4, 6, 8).
- Enter link failure percentage in [0, 100].

## Output and Figures

- A folder named `figures` is created automatically.
- Plots are saved there (e.g., visualization of the topology, average path length vs. failure, switches with multiple failed links, total hosts vs. k).

## Results and Analysis
### 1) Visualization of Fat-Tree Topology
![Fat-Tree Topology Visualization](https://github.com/ItamarMeir/fat-tree-topology-analysis/blob/main/readme_figures/visualization_k4_failure20.0.png?raw=true)

Each color represents a different pod; core switches are gray. Failed links are shown in red.

### 2) Average Path Length vs. Link Failure Percentage
![Average Path Length and No-Path Count vs. Link Failure (k=4)](https://github.com/ItamarMeir/fat-tree-topology-analysis/blob/main/readme_figures/avg_path_length_no_path_count_k4_failure20.0.png?raw=true)

**Important note: Unreachable host pair paths are excluded from average path length calculations.**

As link failure percentage increases, the number of unreachable host pairs increases (no-path count).

We see a rise in average path length as more links fail, but only until a certain point. Beyond that point, many host pairs become unreachable, causing the average path length to drop (because the calculation excludes these pairs).

### 3) Switches with Multiple Failed Links and Ratio vs. Switch Port Count

![Switches with Multiple Failed Links vs. Switch Port Count (at 1% Link Failure)](https://github.com/ItamarMeir/fat-tree-topology-analysis/blob/main/readme_figures/sw_failed_links_vs_k.png?raw=true)

**Notes:**

**1) We consider a switch to have multiple failed links if it has 2 or more failed links.**

**2) The number of switches with multiple failed links is averaged over 100 simulations for each k value. Therefore the y-axis values could be non-integer.**

As the number of ports per switch (k) increases, the number of switches with multiple failed links also increases at a 1% link failure rate.

The overall trend is that the ratio of total switches to switches with multiple failed links decreases with increasing k.

This indicates that larger Fat-Tree topologies are more susceptible to switches experiencing multiple link failures, which can impact network performance and reliability.

### 4) Total Hosts Supported vs. Switch Port Count
![Total Hosts Supported vs. Switch Port Count](https://github.com/ItamarMeir/fat-tree-topology-analysis/blob/main/readme_figures/total_hosts_supported_vs_k.png?raw=true)

The total number of hosts supported in a Fat-Tree topology increases quadratically with the number of ports per switch (k), following the formula: 

Total Hosts = (k^3) / 4.

## What’s in the FatTree Class

Class: `FatTree(k: int, failure_percentage: float = 0.0)`
- Constructs a Fat-Tree graph with hosts and simulates optional link failures.

Key attributes
- k: number of ports per switch (must be even).
- pods: number of pods (= k).
- core_switches: (k/2)^2.
- agg_switches_per_pod: k/2.
- edge_switches_per_pod: k/2.
- hosts_per_edge_switch: k/2.
- failure_percentage: link failure rate used for simulations.
- graph: the current NetworkX Graph (may include failures).
- base_graph: the original, failure-free Graph.
- figures_path: output folder for images.

Main methods (short)
- _build_topology(): builds core/aggregation/edge/host nodes and connects them per Fat-Tree rules.
- _hierarchical_pos(): layered layout (core → aggregation → edge → hosts) grouped by pod for plotting.
- _failure_model(failure_percentage): randomly removes edges according to the given failure rate.
- visualize(): draws the topology, colors nodes by pod (core in a neutral gray), and shows failed links.
- verify_correctness(): asserts expected counts of hosts, edge, and aggregation switches per pod.
- plot_average_path_length_link_failure(): plots average host-to-host path length and no-path count vs. failure rate.
- _avg_path_length_link_failure(), _calculate_avg_path_length(): helpers for path metrics.
- _number_of_sw_failed_links(): computes switches with ≥2 failed links at 1% failure across k values.
- plot_sw_failed_links(): plots switches with multiple failed links vs. k.
- plot_total_hosts_supported(): plots total hosts supported vs. k (= ports per switch).

## Notes
- Python 3.12 is recommended (matches the tested environment).

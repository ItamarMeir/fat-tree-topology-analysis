# Implementation of a Fat-Tree network topology using NetworkX
# Realaization of three-tier fat-tree architecture

import networkx as nx
import matplotlib.pyplot as plt
import random
from matplotlib.lines import Line2D
from pathlib import Path

class FatTree:
	# Initialize the three-tier (Core, Aggregation, Edge) hierarchical Fat-Tree topology with parameter k
	def __init__(self, k, failure_percentage=0.0):
		# All values of each tier are derived from k and mentioned in the paper
		self.k = k  # Number of ports per switch
		self.pods = k  # Number of pods
		self.core_switches = (k // 2) ** 2
		self.agg_switches_per_pod = k // 2
		self.edge_switches_per_pod = k // 2
		self.hosts_per_edge_switch = k // 2
		self.failure_percentage = failure_percentage
		self.graph = nx.Graph()
		self._build_topology()
		self.verify_correctness()
		self.base_graph = self.graph.copy() # Keep a copy of the original graph for failure simulations
		self._failure_model(failure_percentage)
		
		# Ensure the figures directory exists
		self.figures_path = Path("figures")
		self.figures_path.mkdir(parents=True, exist_ok=True)

	def _build_topology(self):
		# Create core switches
		core_switch_ids = []
		for i in range(self.core_switches):
			switch_id = f'c{i}'
			core_switch_ids.append(switch_id)
			self.graph.add_node(switch_id, type='core')

		# Create pods
		for pod in range(self.pods):
			agg_switch_ids = []
			edge_switch_ids = []

			# Create aggregation switches
			for i in range(self.agg_switches_per_pod):
				switch_id = f'a{pod}_{i}'
				agg_switch_ids.append(switch_id)
				self.graph.add_node(switch_id, type='aggregation', pod=pod)

			# Create edge switches and hosts
			for i in range(self.edge_switches_per_pod):
				edge_switch_id = f'e{pod}_{i}'
				edge_switch_ids.append(edge_switch_id)
				self.graph.add_node(edge_switch_id, type='edge', pod=pod)

				# Connect edge switch to hosts
				for j in range(self.hosts_per_edge_switch):
					host_id = f'h{pod}_{i}_{j}'
					self.graph.add_node(host_id, type='host', pod=pod)
					self.graph.add_edge(edge_switch_id, host_id)

			# Connect aggregation switches to edge switches
			for agg_index, agg_switch in enumerate(agg_switch_ids):
				for edge_index, edge_switch in enumerate(edge_switch_ids):
					self.graph.add_edge(agg_switch, edge_switch)

			# Connect aggregation switches to core switches
			for agg_index, agg_switch in enumerate(agg_switch_ids):
				for core_index in range(self.core_switches):
					if core_index // (self.k // 2) == agg_index:
						core_switch = core_switch_ids[core_index]
						self.graph.add_edge(agg_switch, core_switch)

	def _hierarchical_pos(self):
		# Create a layered layout grouped by pod: core (top), aggregation, edge, hosts (bottom)
		types = nx.get_node_attributes(self.graph, 'type')
		pos = {}
		layers = ['core', 'aggregation', 'edge', 'host']
		for level, layer in enumerate(layers):
			y = len(layers) - 1 - level
			if layer == 'core':
				nodes = sorted([n for n, t in types.items() if t == 'core'])
				n = len(nodes)
				for i, node in enumerate(nodes):
					x = (i + 1) / (n + 1) if n else 0.5
					pos[node] = (x, y)
				continue

			nodes = [n for n, t in types.items() if t == layer]
			if not nodes:
				continue
			pods = sorted({self.graph.nodes[n]['pod'] for n in nodes})
			seg_w = 1.0 / max(1, len(pods))
			for p_idx, p in enumerate(pods):
				group = sorted([n for n in nodes if self.graph.nodes[n]['pod'] == p])
				m = len(group)
				for i, node in enumerate(group):
					x = p_idx * seg_w + (i + 1) / (m + 1) * seg_w
					pos[node] = (x, y)
		return pos

	def _failure_model(self, failure_percentage):
		# Simulate random failures of links based on the given failure rate
		self.graph = self.base_graph.copy()  # Reset to original graph before applying failures
		failed_links = []
		for u, v in self.graph.edges():
			if random.random() < failure_percentage / 100:
				failed_links.append((u, v))
		self.graph.remove_edges_from(failed_links)
		return failed_links

	def visualize(self):
			# Visualize the Fat-Tree topology using Matplotlib
			print(f"Visualizing Fat-Tree Topology with k={self.k} and link failure rate of {self.failure_percentage}%")
			pos = self._hierarchical_pos()

			# Determine failed links by comparing base_graph (original) with current graph
			original_edges = set(self.base_graph.edges())
			current_edges = set(self.graph.edges())
			failed_edges = list(original_edges - current_edges)
			n_failed = len(failed_edges)

			# Color by pod: all nodes with the same pod share a color; core (no pod) gets a neutral color
			pods_attr = nx.get_node_attributes(self.graph, 'pod')
			pods = sorted({pods_attr[n] for n in pods_attr})
			cmap = plt.get_cmap('tab20')
			pod_color_map = {p: cmap(i % 20) for i, p in enumerate(pods)}
			core_color = 'dimgray'

			node_colors = []
			for node in self.graph.nodes():
				pod = self.graph.nodes[node].get('pod', None)
				if pod is None:
					node_colors.append(core_color)
				else:
					node_colors.append(pod_color_map[pod])

			# Draw nodes and labels
			nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500)
			nx.draw_networkx_labels(self.graph, pos, font_size=8)

			# Draw existing (up) edges uniformly
			nx.draw_networkx_edges(self.graph, pos, width=1.5, edge_color='lightgray', alpha=0.8)

			# Draw failed edges (from original topology) as dashed red lines
			if failed_edges:
				nx.draw_networkx_edges(self.base_graph, pos, edgelist=failed_edges,
									   style='dashed', edge_color='red', alpha=0.9, width=2)

			# Title showing k and number of failed links
			plt.title(f"Fat-Tree Topology (k={self.k}) — failed links: {n_failed}, failure rate: {self.failure_percentage}%", fontsize=14)

			# Legend for pods, core and failed links
			try:
				import matplotlib.patches as mpatches
				handles = [mpatches.Patch(color=core_color, label='core')]
				for p in pods:
					handles.append(mpatches.Patch(color=pod_color_map[p], label=f'pod {p}'))
				if n_failed:
					handles.append(Line2D([0], [0], color='red', lw=2, linestyle='--', label=f'failed links ({n_failed})'))
				plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05),
					ncol=min(len(handles), 6), fontsize=8, frameon=False)
			except Exception:
				pass

			plt.tight_layout()
		
			# Save the plot as an image file (workspace/figures/visualization_k{self.k}_failure{self.failure_percentage}.png)
			plt.savefig(f"{self.figures_path}/visualization_k{self.k}_failure{self.failure_percentage}.png")
			print(f"Figure saved as {self.figures_path}/visualization_k{self.k}_failure{self.failure_percentage}.png")

			# Show the plot
			plt.show()
			# Close the plot to free up resources
			plt.close()

	def verify_correctness(self):
		# Verify the correctness of the Fat-Tree topology
		expected_hosts = (self.k ** 3) // 4
		actual_hosts = len([n for n, t in nx.get_node_attributes(self.graph, 'type').items() if t == 'host'])
		assert expected_hosts == actual_hosts, f"Expected {expected_hosts} hosts, found {actual_hosts}."

		# Verify number of hosts, edges, and aggregate switches per pod:
		expected_hosts_per_pod = (self.k ** 2) // 4
		expected_edges_per_pod = expected_agg_in_pod = self.k // 2
		for pod in range(self.pods):
			actual_hosts_in_pod = len([n for n, t in nx.get_node_attributes(self.graph, 'type').items()
				if t == 'host' and self.graph.nodes[n]['pod'] == pod])
			actual_edges_in_pod = len([n for n, t in nx.get_node_attributes(self.graph, 'type').items()
				if t == 'edge' and self.graph.nodes[n]['pod'] == pod])
			actual_agg_in_pod = len([n for n, t in nx.get_node_attributes(self.graph, 'type').items()
				if t == 'aggregation' and self.graph.nodes[n]['pod'] == pod])
			assert expected_agg_in_pod == actual_agg_in_pod, \
				f"Pod {pod}: Expected {expected_agg_in_pod} aggregate switches, found {actual_agg_in_pod}."
			assert expected_edges_per_pod == actual_edges_in_pod, \
				f"Pod {pod}: Expected {expected_edges_per_pod} edges, found {actual_edges_in_pod}."
			assert expected_hosts_per_pod == actual_hosts_in_pod, \
				f"Pod {pod}: Expected {expected_hosts_per_pod} hosts, found {actual_hosts_in_pod}."
	
	def plot_average_path_length_link_failure(self):
		# Plot average path length vs. link failure percentage and no-path count on secondary y-axis
		print(f"Plotting Average Path Length and No-Path Count vs. Link Failure Percentage for k={self.k}")
		failure_rates = list(range(0, 101, 10))
		avg_path_lengths = []
		no_path_counts = []

		for rate in failure_rates:
			self._failure_model(rate)
			avg_length, no_path_count = self._avg_path_length_link_failure(rate)
			avg_path_lengths.append(avg_length)
			no_path_counts.append(no_path_count)

		fig, ax1 = plt.subplots(figsize=(10, 6))

		# Plot average path length on primary y-axis
		line1, = ax1.plot(failure_rates, avg_path_lengths, marker='o', label='Average Path Length', color='blue')
		ax1.set_xlabel('Link Failure Percentage (%)')
		ax1.set_ylabel('Average Path Length', color='blue')
		ax1.tick_params(axis='y', labelcolor='blue')
		ax1.set_xticks(failure_rates)
		ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.3)

		# Set y-limits for average path length if finite values exist
		finite_lengths = [x for x in avg_path_lengths if x != float('inf')]
		if finite_lengths:
			ax1.set_ylim(0, max(finite_lengths) * 1.1)

		# Plot no-path counts on secondary y-axis
		ax2 = ax1.twinx()
		line2, = ax2.plot(failure_rates, no_path_counts, marker='s', label='No-Path Count', color='red')
		ax2.set_ylabel('No-Path Count', color='red')
		ax2.tick_params(axis='y', labelcolor='red')

		# Combine legends from both axes
		lines = [line1, line2]
		labels = [l.get_label() for l in lines]
		ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)

		plt.title(f'Average Path Length and No-Path Count vs. Link Failure Percentage (k={self.k})')
		plt.tight_layout()

		# Save the figure
		plt.savefig(f"{self.figures_path}/avg_path_length_no_path_count_k{self.k}_failure{failure_percentage}.png")
		print(f"Figure saved as {self.figures_path}/avg_path_length_no_path_count_k{self.k}_failure{failure_percentage}.png")

		# Show the plot
		plt.show()
		# Close the plot to free up resources
		plt.close()

	def _avg_path_length_link_failure(self, failure_percentage):
		# Calculate the average path length between all pairs of hosts considering link failures
		# Calculate over a fixed number of graphs (due to randomness in failures)
		num_graphs = 10
		total_path_length = 0
		total_no_path_count = 0
		for _ in range(num_graphs):
			self._failure_model(failure_percentage)
			path_length, no_path_count = self._calculate_avg_path_length()
			total_path_length += 0 if path_length == float('inf') else path_length
			total_no_path_count += no_path_count

		avg_path_length = total_path_length / num_graphs if num_graphs > 0 else float('inf')
		avg_no_path_count = total_no_path_count // num_graphs if num_graphs > 0 else 0

		return avg_path_length, avg_no_path_count

	def _calculate_avg_path_length(self):
		# Calculate average path length between all pairs of hosts in the current graph
		host_nodes = [n for n, t in nx.get_node_attributes(self.graph, 'type').items() if t == 'host']
		total_length = 0
		count = 0
		no_path_count = 0
		for i in range(len(host_nodes)):
			for j in range(i + 1, len(host_nodes)):
				try:
					length = nx.shortest_path_length(self.graph, source=host_nodes[i], target=host_nodes[j])
					total_length += length
					count += 1
				except nx.NetworkXNoPath:
					no_path_count += 1
					length = float('inf')
					continue
		return total_length / count if count > 0 else float('inf'), no_path_count
	
	def _number_of_sw_failed_links(self):
		# Number of switches with multiple (2 or more) failed links (at a 1% failure rate) as a function of switch port count
		num_of_graphs = 100   # Average over multiple graphs for randomness
		k_values = [4, 6, 8, 10, 12, 14, 16]
		sw_failed_links = []
		for k in k_values:
			fat_tree = FatTree(k)
			total_multiple_failed = 0
			for _ in range(num_of_graphs):
				fat_tree._failure_model(1.0)  # 1% failure rate
				switch_failed_link_count = {}
				for u, v in fat_tree.base_graph.edges():
					if not fat_tree.graph.has_edge(u, v):
						if fat_tree.graph.nodes[u]['type'] != 'host':
							switch_failed_link_count[u] = switch_failed_link_count.get(u, 0) + 1
						if fat_tree.graph.nodes[v]['type'] != 'host':
							switch_failed_link_count[v] = switch_failed_link_count.get(v, 0) + 1
				num_switches_multiple_failed_links = sum(1 for count in switch_failed_link_count.values() if count >= 2)
				total_multiple_failed += num_switches_multiple_failed_links
			# average number of switches with >=2 failed links for this k
			avg_multiple_failed = total_multiple_failed / num_of_graphs
			sw_failed_links.append(avg_multiple_failed)
			
		return k_values, sw_failed_links # x,y values for plotting
	
	def plot_sw_failed_links(self):
		# Plot number of switches with multiple failed links vs. switch port count
		print("Plotting Switches with Multiple Failed Links vs. Switch Port Count (=k)")
		k_values, sw_failed_links = self._number_of_sw_failed_links()

		# Primary axis: number of switches with >=2 failed links
		fig, ax1 = plt.subplots(figsize=(10, 6))
		line1, = ax1.plot(k_values, sw_failed_links, marker='o', color='green', label='Switches with ≥2 Failed Links')
		ax1.set_xlabel('Switch Port Count (=k)')
		ax1.set_ylabel('Number of Switches with ≥2 Failed Links (at 1% failure rate)', color='green')
		ax1.tick_params(axis='y', labelcolor='green')
		ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.3)

		# Secondary axis: ratio (total switches (=k^3/4)) / (switches with multiple failed links)
		ratios = []
		for k, failed in zip(k_values, sw_failed_links):
			total_switches = (k ** 3) / 4  # as requested
			ratios.append(total_switches / failed if failed > 0 else float('inf'))

		ax2 = ax1.twinx()
		line2, = ax2.plot(k_values, ratios, marker='^', color='orange',
						  label='Ratio: (Total Switches) / ( Switches with ≥2 Failed Links)')
		ax2.set_ylabel('Ratio: Total Switches / Switches with ≥2 Failed Links', color='orange')
		ax2.tick_params(axis='y', labelcolor='orange')

		# Combine legends
		lines = [line1, line2]
		labels = [l.get_label() for l in lines]
		ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=False)

		plt.title('Switches with Multiple Failed Links and Ratio vs. Switch Port Count (=k)')
		plt.tight_layout()

		# Save the figure
		plt.savefig(f"{self.figures_path}/sw_failed_links_vs_k.png")
		print(f"Figure saved as {self.figures_path}/sw_failed_links_vs_k.png")

		# Show the plot
		plt.show()
		# Close the plot to free up resources
		plt.close()

	def plot_total_hosts_supported(self):
		print("Plotting Total Hosts Supported vs. Number of Port Count (=k)")
		# Plot the total number of hosts supported in the Fat-Tree topology as a function of k
		k_values = list(range(2, 128, 2))  # Even k values from 2 to 128
		host_counts = [(k ** 3) // 4 for k in k_values]
		# Plotting host counts vs. number of ports
		plt.figure(figsize=(10, 6))
		plt.plot(k_values, host_counts, marker='o', color='purple')
		plt.xlabel('Number of Port Count')
		plt.ylabel('Total Number of Hosts Supported')
		plt.title('Total Hosts Supported vs. Number of Port Count (=k) in Fat-Tree Topology')
		plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.3)
		plt.tight_layout()

		# Save the figure
		plt.savefig(f"{self.figures_path}/total_hosts_supported_vs_k.png")
		print(f"Figure saved as {self.figures_path}/total_hosts_supported_vs_k.png")

		# Show the plot
		plt.show()
		# Close the plot to free up resources
		plt.close()


if __name__ == "__main__":
	k = int(input("Enter the value of k (must be even): "))
	failure_percentage = float(input("Enter the link failure percentage (0 to 100): "))
	if k % 2 != 0 or not (0 <= failure_percentage <= 100):
		raise ValueError("k must be an even integer.")
	fat_tree = FatTree(k, failure_percentage)
	fat_tree.visualize()
	fat_tree.plot_average_path_length_link_failure()
	fat_tree.plot_sw_failed_links()
	fat_tree.plot_total_hosts_supported()
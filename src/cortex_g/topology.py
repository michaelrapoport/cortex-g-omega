from collections import defaultdict
import torch

class TopologyManager:
    """
    Manages the 'liquid' graph structure, including synapse creation,
    pruning, and Hebbian learning.
    """
    def __init__(self):
        # Map: Source_ID -> List[ (Target_ID, Edge_Operator_Vector) ]
        self.synapses = defaultdict(list)
        
        # Map: (Source_ID, Target_ID) -> health_score
        self.edge_health = defaultdict(lambda: 1.0)

    def add_synapse(self, src_idx, dst_idx, edge_operator):
        """Creates a new connection between two nodes."""
        self.synapses[src_idx].append((dst_idx, edge_operator))
        self.edge_health[(src_idx, dst_idx)] = 1.0 

    def structural_plasticity(self, src_idx, dst_idx):
        """
        Rewrites the graph topology based on signal flux.
        "Cells that fire together, wire together."
        """
        edge_key = (src_idx, dst_idx)
        self.edge_health[edge_key] += 0.05
        
        if self.edge_health[edge_key] > 50.0:
             self.edge_health[edge_key] = 50.0
             
        # Wormhole condition (Long-term Potentiation)
        if self.edge_health[edge_key] > 40.0:
            self._create_wormhole(src_idx, dst_idx)

    def _create_wormhole(self, src_idx, dst_idx):
        """Optimization: Creates direct shortcuts for high-traffic paths."""
        if dst_idx in self.synapses:
            for grand_dst_idx, _ in self.synapses[dst_idx]:
                if self.edge_health.get((dst_idx, grand_dst_idx), 0) > 40.0:
                    # Check if shortcut already exists
                    current_targets = [t[0] for t in self.synapses[src_idx]]
                    if grand_dst_idx not in current_targets:
                        # Create new random operator (simulating composite bind)
                        new_op = torch.randint(0, 2, (156,), dtype=torch.int64)
                        self.add_synapse(src_idx, grand_dst_idx, new_op)

    def prune_dead_synapses(self):
        """Garbage collection for atrophied logic pathways."""
        to_remove = []
        for edge_key, health in self.edge_health.items():
            self.edge_health[edge_key] -= 0.001 # Decay
            if self.edge_health[edge_key] < 0:
                to_remove.append(edge_key)
                
        for src, dst in to_remove:
            del self.edge_health[(src, dst)]
            if src in self.synapses:
                self.synapses[src] = [t for t in self.synapses[src] if t[0] != dst]
import torch
import torch.nn as nn
from .algebra import HDC_Algebra
from .topology import TopologyManager

class CortexOmega(nn.Module):
    def __init__(self, node_capacity=10000, dim_chunks=156): 
        super().__init__()
        self.capacity = node_capacity
        self.algebra = HDC_Algebra(dim_bits=dim_chunks * 64)
        self.topology = TopologyManager()
        
        # State Tensor: Random Orthogonal Initialization
        self.world_state = torch.randint(
            low=0, high=2, size=(node_capacity, dim_chunks), dtype=torch.int64
        )
        
        # Homeostasis
        self.thresholds = torch.ones(node_capacity) * 0.45 
        self.local_clocks = torch.ones(node_capacity)
        
    def propagate(self, node_idx, input_hologram):
        """
        The Atomic Reasoning Step.
        Returns downstream events if Surprise > Threshold.
        """
        current_belief = self.world_state[node_idx]
        dist = self.algebra.similarity(current_belief, input_hologram)
        
        # Noise Gate
        if dist < self.thresholds[node_idx]: 
            self.thresholds[node_idx] *= 0.999 # Increase sensitivity (Curiosity)
            return None 
            
        # Holographic Integration
        update_mask = torch.bitwise_xor(current_belief, input_hologram)
        self.world_state[node_idx] = torch.bitwise_xor(current_belief, update_mask)
        
        # Adaptivity
        self.thresholds[node_idx] += (dist - self.thresholds[node_idx]) * 0.1
        self.local_clocks[node_idx] = 1.0 + (dist * 5.0) # Time Dilation
        
        # Propagation
        downstream_spikes = []
        if node_idx in self.topology.synapses:
            for target, edge_op in self.topology.synapses[node_idx]:
                rotated_signal = self.algebra.bind(input_hologram, edge_op)
                self.topology.structural_plasticity(node_idx, target)
                downstream_spikes.append((target, rotated_signal))
                
        return downstream_spikes
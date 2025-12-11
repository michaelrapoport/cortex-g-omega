import torch
import random
from cortex_g.kernel import CortexOmega
from cortex_g.engine import SimulationEngine

def main():
    print("Initializing Cortex-G Omega Kernel...")
    
    # 1. Setup Kernel
    kernel = CortexOmega(node_capacity=100)
    
    # 2. Build a random graph topology for testing
    print("Building Random Liquid Topology...")
    for i in range(100):
        # Connect to 3 random neighbors
        targets = random.sample(range(100), 3)
        for t in targets:
            if t != i:
                # Random Edge Operator
                op = torch.randint(0, 2, (156,), dtype=torch.int64)
                kernel.topology.add_synapse(i, t, op)
                
    # 3. Create Simulation Engine
    sim = SimulationEngine(kernel)
    
    # 4. Inject a "Thought"
    print("Injecting Semantic Vector...")
    seed_vector = torch.randint(0, 2, (156,), dtype=torch.int64)
    sim.inject_event(target_node=0, vector=seed_vector)
    
    # 5. Run
    sim.run(max_duration=20.0)

if __name__ == "__main__":
    main()
import heapq

class SimulationEngine:
    """
    Manages the asynchronous, relativistic event loop.
    """
    def __init__(self, kernel):
        self.kernel = kernel
        self.event_queue = [] # Priority Queue
        self.event_id_counter = 0
        self.current_time = 0.0

    def inject_event(self, target_node, vector, delay=0.0):
        exec_time = self.current_time + delay
        # Tuple: (Time, ID, Target, Vector)
        heapq.heappush(self.event_queue, (exec_time, self.event_id_counter, target_node, vector))
        self.event_id_counter += 1

    def run(self, max_duration=100.0, verbose=True):
        if verbose: print(f"--- Cortex-G Simulation Started ---")
        
        while self.event_queue and self.current_time < max_duration:
            exec_time, _, target_node, vector = heapq.heappop(self.event_queue)
            self.current_time = exec_time
            
            # Step the Kernel
            downstream = self.kernel.propagate(target_node, vector)
            
            if downstream:
                if verbose: print(f"T={self.current_time:.4f} | Node {target_node} Fired -> {len(downstream)} targets")
                for next_target, next_vector in downstream:
                    # Relativistic Delay: 1 / Local_Clock_Rate
                    delay = 1.0 / self.kernel.local_clocks[next_target].item()
                    self.inject_event(next_target, next_vector, delay)

            # Periodic Topology Maintenance
            if self.event_id_counter % 100 == 0:
                self.kernel.topology.prune_dead_synapses()
        
        status = "Converged" if not self.event_queue else "Time Limit Reached"
        if verbose: print(f"--- {status} at T={self.current_time:.4f} ---")
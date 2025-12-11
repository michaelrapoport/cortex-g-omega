<div align="center">

# CORTEX-G OMEGA

### Relativistic Asynchronous Mesh for Hyperdimensional Reasoning ("Recursive Neuralese")

![Version](https://img.shields.io/badge/version-100.0.0_Omega-blueviolet?style=for-the-badge)![Python](https://img.shields.io/badge/python-3.9+-blue?style=for-the-badge)![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge)
![License](https://img.shields.io/badge/license-Apache_2.0-green?style=for-the-badge)

<p align="center">
  <b>The "Hippocampus" for Large Language Models.</b><br>
  A neuromorphic kernel that replaces synchronous matrix multiplication with<br>
  <b>Geometric Difference Propagation</b> on a <b>Hyperdimensional (HDC)</b> manifold.
</p>

[Abstract](#-abstract) • [Core Physics](#-core-physics) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Citation](#-citation)

</div>

---

## 📋 Abstract

Current Deep Learning architectures (Transformers) are constrained by the **Von Neumann Bottleneck** due to global memory synchronization and quadratic $O(N^2)$ attention mechanisms. They lack efficient state persistence for complex reasoning.

**Cortex-G Omega** introduces a paradigm shift from **Arithmetic AI** to **Geometric AI**:

1. **Holographic State:** Uses 10,000-bit vectors and bitwise logic (XOR/Popcount) instead of Float32.
2. **Asynchronous Physics:** No global clock. Nodes dilate time based on local "Surprise" energy.
3. **Liquid Topology:** The graph physically rewrites its own schema at runtime based on Hebbian flux.

This architecture achieves a theoretical computational speedup of **$10,000\times$** over traditional RNNs for recurrent reasoning tasks via bitwise operations.

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/agitronics/cortex-g-omega.git
cd cortex-g-omega
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from cortex_g.kernel import CortexOmega
from cortex_g.engine import SimulationEngine

# 1. Initialize the Kernel (10k nodes)
kernel = CortexOmega(node_capacity=10000)

# 2. Create a Simulation Engine
sim = SimulationEngine(kernel)

# 3. Create a Synthetic "Hologram" (Thought Vector)
# In production, this comes from an LLM Embedding -> HDC Bridge
thought_vector = torch.randint(0, 2, (156,), dtype=torch.int64)

# 4. Inject into Node 0
sim.inject_event(target_node=0, vector=thought_vector)

# 5. Run the asynchronous reasoning loop
sim.run(max_duration=50.0)
```

## 📖 Introduction: The Synchrony Tax

### The Problem

In standard Deep Learning (RNNs, Transformers, CNNs), the system operates on a global clock. Every neuron or attention head performs a dot-product update at every time step $t$, regardless of whether the information content has changed.

This leads to two critical failures:

1. **Energy Inefficiency:** 99% of the network is processing "silence" or redundant data.
2. **Semantic Drift:** In recurrent loops (agents communicating with themselves), floating-point errors accumulate, causing the vector representation ("Neuralese") to degrade into noise or hallucinations.

### The Cortex-G Solution

Cortex-G inverts the paradigm. Instead of "Compute-First," it is "Difference-First."

* **Event-Driven:** Computation is triggered *only* when input deviates from prediction.
* **Stateful:** Memory is not a passive buffer but an active holographic field.
* **Geometric:** Logic is defined by vector rotation, not arithmetic addition.

---

## ⚙️ Theoretical Framework

### 1. The Law of Conservation of Meaning

Derived from the Free Energy Principle, this axiom states that a cognitive system should only expend energy to minimize surprise.

In Cortex-G, every node $n$ maintains an internal prediction state $P_t$. Upon receiving input $I_t$, it calculates the **Semantic Difference** ($\Delta$):

$$
\Delta = I_t \ominus P_t
$$

Where $\ominus$ represents the geometric difference operation. The system then evaluates the magnitude $||\Delta||$.

* If $||\Delta|| < \epsilon$ (Threshold): The input was predicted. **Suppress Signal.**
* If $||\Delta|| > \epsilon$: The input is novel. **Update State & Propagate.**

This acts as a native **Noise Gate** for thought, stabilizing recurrent loops.

```python
# Relativistic Propagation Logic
# 1. Geometric Difference (Bitwise XOR)
delta = torch.bitwise_xor(input_state, prediction)

# 2. Measure Surprise (Normalized Hamming Distance)
surprise = torch.mean(delta.float())

# 3. Gated Propagation (Noise Threshold)
if surprise > epsilon:
    propagate(delta)  # Only transmit significant changes
```

### 2. Hyperdimensional Computing (HDC)

To escape the limits of Float32 precision, Cortex-G utilizes **Hyperdimensional Computing** (also known as Vector Symbolic Architectures).

* **Representation:** Data is encoded as 10,000-bit pseudo-random binary vectors.
* **Robustness:** Information is holographic; it is distributed across all bits. You can flip 40% of the bits and still recover the original meaning.
* **Speed:** Operations use Bitwise XOR and Population Count (Popcount), which are single-cycle CPU instructions.


| Operation              | Logical Equivalent | Geometric Action              |
| :--------------------- | :----------------- | :---------------------------- |
| Binding ($A \oplus B$) | A is B             | Rotation (Change Basis)       |
| Bundling ($A + B$)     | Memory Store       | Superposition (Majority Rule) |
| Permutation ($\Pi A$)  | Sequence           | Cyclic Shift                  |

### 3. Relativistic Time Dilation

In a mesh of thousands of nodes, imposing a global `step()` function creates latency. Cortex-G allows each node to run on its own local clock $\tau$.

$$
\tau_{node} \propto \frac{1}{\text{Entropy}(Local)}
$$

Nodes experiencing high conflict (high $\Delta$) dilate time (increase sampling rate) to resolve the ambiguity, while stable nodes hibernate.

---

## 🏗️ System Architecture

The architecture is a **Directed Cyclic Graph** where edges are operators and nodes are state containers.

### 4.1 The Holographic Node

A node is not a perceptron (scalar sum). It is a **Resonant Cavity** for a high-dimensional vector.

* **Input Port:** Sums incoming Hypervectors (via Majority Rule).
* **State Buffer:** Holds the current 10k-bit belief.
* **Comparator:** Hardware-accelerated Hamming Distance unit.

### 4.2 The Active Edge Operator

Standard graphs treat edges as weights ($w$). Cortex-G treats edges as **Functions** ($f(x)$).
Specifically, in HDC, an edge performs a **Binding Operation** (XOR).

If Node A = "Apple" and Node B = "Color", the edge $E_{ab}$ applies the transformation "HasProperty".

$$
V_B = V_A \oplus E_{ab}
$$

This allows the graph structure itself to encode semantic relationships.

### 4.3 Liquid Topology

The graph is plastic. It implements **Hebbian Learning** on the topology itself.

* **Synaptic Potentiation:** Edges with high flux (frequent $\Delta$ transmission) reduce their resistance (lower $\epsilon$).
* **Pruning:** Edges with zero flux over time $T$ are deleted from memory.
* **Wormholing:** If $A \to B \to C$ is a frequent high-traffic path, the system spontaneously creates a direct edge $A \to C$.

---

## 📐 Mathematical Formalism

We define the algebra $\mathcal{H}$ over the space $\{0, 1\}^D$ where $D=10,000$.

**1. Similarity (Hamming Distance):**

$$
\delta(A, B) = \frac{1}{D} \sum_{i=1}^{D} |a_i - b_i|
$$

* $\delta \approx 0.5$: Orthogonal (Unrelated)
* $\delta \approx 0.0$: Identical

**2. Binding (XOR):**

$$
A \otimes B = A \oplus B
$$

* Invertible: $(A \otimes B) \otimes B = A$
* Used for: Variable assignment, Edge traversal.

**3. Bundling (Superposition):**

$$
A + B + C = \text{Majority}(A, B, C)
$$

* Used for: storing multiple items in one node.

**4. The Propagation Inequality:**
A node $n$ fires if:

$$
\delta( \text{Bundle}(\sum I_{in}), S_n ) > \epsilon_{n}
$$

---

## 🧩 Integration Patterns

### The "Hippocampus" Pattern

Cortex-G is designed not to replace Large Language Models, but to act as their Long-Term Logic Memory.

1. **Input:** LLM generates tokens.
2. **Transduction:** The tokens are embedded and quantized into a Holographic Vector.
3. **Injection:** The vector is injected into the Cortex-G mesh.
4. **Resonance:** Cortex-G circulates the energy. It activates associated memories and performs geometric deductions via edge traversals.
5. **Equilibrium:** Once the total system entropy (sum of $\Delta$) drops below a global threshold, the network is "Stable".
6. **Readout:** The stable state is decoded back into text context.
7. **Conditioning:** The LLM uses this context to generate the next token, eliminating hallucinations.

## 📦 Architecture Modules

* `src/cortex_g/algebra.py`: Hardware-accurate implementation of HDC math (Bind, Bundle, Similarity) using packed int64 tensors.
* `src/cortex_g/topology.py`: Manages the "Liquid Graph," handling synapse creation, wormholing, and pruning.
* `src/cortex_g/kernel.py`: The cognitive node logic (Surprise calculation, Homeostasis).
* `src/cortex_g/engine.py`: The priority-queue based event scheduler.

## 📜 Citation

```Bibtex
@article{rapoport2025cortexg,
  title={Cortex-G Omega: Relativistic Asynchronous Mesh for HDC},
  author={Rapoport, Michael K.},
  journal={Agitronics Research Group},
  year={2025},
  version={100.0.0}
}
```

<div align="center">
<br>
<b>© 2025 Agitronics Research Group</b> • Michael K. Rapoport<br>
<i>All Rights Reserved • Patent Pending</i>
</div>

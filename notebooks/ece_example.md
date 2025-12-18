# Envy-Cycle Elimination (ECE) Algorithm: A Detailed Example

This document provides a step-by-step walkthrough of the Envy-Cycle Elimination algorithm for fair division, demonstrating how it achieves an EF1 (envy-free up to one item) allocation. The example matches the implementation in `eval_pipeline/utils/inference.py`.

## Table of Contents
1. [Introduction](#introduction)
2. [Algorithm Overview](#algorithm-overview)
3. [Example Setup](#example-setup)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Final Allocation](#final-allocation)
6. [EF1 Verification](#ef1-verification)
7. [Key Observations](#key-observations)

---

## Introduction

**Envy-Cycle Elimination (ECE)** is a fundamental algorithm in fair division theory. Given a set of agents and indivisible items, ECE produces an allocation that is **EF1 (Envy-Free up to One item)** — meaning any envy an agent feels toward another can be eliminated by removing a single item from the envied bundle.

### Key Properties
- **Polynomial time**: O(nm²) where n = agents, m = items
- **EF1 guarantee**: Always produces an EF1 allocation
- **Online compatible**: Items can arrive one at a time

---

## Algorithm Overview

The ECE algorithm processes items one at a time:

```
For each item g:
    1. Build the "envy graph" based on current bundles
       - Nodes = agents
       - Edge A → B means agent A envies agent B

    2. While there exists an envy cycle:
       - Rotate bundles along the cycle
       - (Each agent in the cycle gets the bundle of the agent they envy)

    3. Find an "unenvied" agent (no incoming edges)
       - Such an agent always exists after cycle elimination

    4. Give item g to the unenvied agent (smallest index for determinism)
```

### Why Cycles Must Be Eliminated

If there's a cycle (e.g., A→B→A), then every agent in the cycle is envied by someone, so there may be no "unenvied" agent. By rotating bundles along the cycle, we break the cycle without making anyone worse off (each agent gets a bundle they value at least as much).

### Tie-Breaking Rule

When multiple agents are unenvied, the implementation selects the agent with the **smallest index** for determinism.

---

## Example Setup

### Agents
- **Alice (A)** - Agent 0
- **Bob (B)** - Agent 1
- **Charlie (C)** - Agent 2

### Valuation Matrix

Each agent has a value for each item (higher = more preferred):

| Agent   | Item 1 | Item 2 | Item 3 | Item 4 | Item 5 | Item 6 | Item 7 | Item 8 | Item 9 | Item 10 |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|
| Alice   | 8      | 3      | 7      | 2      | 9      | 4      | 6      | 1      | 5      | 10      |
| Bob     | 5      | 9      | 2      | 8      | 3      | 7      | 1      | 6      | 10     | 4       |
| Charlie | 3      | 6      | 10     | 5      | 2      | 8      | 9      | 4      | 1      | 7       |

### Notation
- **Bundle**: The set of items an agent currently holds
- **v_A(S)**: Alice's value for bundle S (sum of item values)
- **A → B**: Alice envies Bob (Alice values Bob's bundle more than her own)

---

## Step-by-Step Walkthrough

### Initial State

All bundles are empty:
- Alice's bundle: {} → value = 0
- Bob's bundle: {} → value = 0
- Charlie's bundle: {} → value = 0

---

### Round 1: Allocating Item 1

**Item 1 values**: Alice=8, Bob=5, Charlie=3

**Current bundles:**
| Agent   | Bundle | Own Value |
|---------|--------|-----------|
| Alice   | {}     | 0         |
| Bob     | {}     | 0         |
| Charlie | {}     | 0         |

**Envy Analysis:**
- Alice: No envy (all bundles empty)
- Bob: No envy
- Charlie: No envy

**Envy Graph:**
```
    A       B       C
    (no edges - no one envies anyone)
```

**Unenvied agents**: Alice, Bob, Charlie

**Action**: Give Item 1 to Alice (min index = 0)

**Updated Bundles:**
- Alice: {1} → value = 8
- Bob: {} → value = 0
- Charlie: {} → value = 0

---

### Round 2: Allocating Item 2

**Item 2 values**: Alice=3, Bob=9, Charlie=6

**Bundle values (row = agent, column = whose bundle they're valuing):**
| Agent   | Values Alice's {1} | Values Bob's {} | Values Charlie's {} |
|---------|--------------------|-----------------|---------------------|
| Alice   | 8                  | 0               | 0                   |
| Bob     | 5                  | 0               | 0                   |
| Charlie | 3                  | 0               | 0                   |

**Envy Analysis:**
- Alice: 8 ≥ 0, 8 ≥ 0 → No envy
- Bob: 0 < 5 → **Bob envies Alice**
- Charlie: 0 < 3 → **Charlie envies Alice**

**Envy Graph:**
```
        A
       ↗ ↖
      B   C
```

**Unenvied agents**: Bob, Charlie (Alice is envied)

**Action**: Give Item 2 to Bob (min index among unenvied = 1)

**Updated Bundles:**
- Alice: {1} → value = 8
- Bob: {2} → value = 9
- Charlie: {} → value = 0

---

### Round 3: Allocating Item 3

**Item 3 values**: Alice=7, Bob=2, Charlie=10

**Bundle values:**
| Agent   | Values A's {1} | Values B's {2} | Values C's {} |
|---------|----------------|----------------|---------------|
| Alice   | 8              | 3              | 0             |
| Bob     | 5              | 9              | 0             |
| Charlie | 3              | 6              | 0             |

**Envy Analysis:**
- Alice: 8 ≥ 3, 8 ≥ 0 → No envy
- Bob: 9 ≥ 5, 9 ≥ 0 → No envy
- Charlie: 0 < 3 (envies A), 0 < 6 (envies B) → **Charlie envies both**

**Envy Graph:**
```
    A ←─── C ───→ B
```

**Unenvied agents**: Charlie (only one not envied)

**Action**: Give Item 3 to Charlie

**Updated Bundles:**
- Alice: {1} → value = 8
- Bob: {2} → value = 9
- Charlie: {3} → value = 10

---

### Round 4: Allocating Item 4

**Item 4 values**: Alice=2, Bob=8, Charlie=5

**Bundle values:**
| Agent   | Values A's {1} | Values B's {2} | Values C's {3} |
|---------|----------------|----------------|----------------|
| Alice   | 8              | 3              | 7              |
| Bob     | 5              | 9              | 2              |
| Charlie | 3              | 6              | 10             |

**Envy Analysis:**
- Alice: 8 ≥ 3, 8 ≥ 7 → No envy
- Bob: 9 ≥ 5, 9 ≥ 2 → No envy
- Charlie: 10 ≥ 3, 10 ≥ 6 → No envy

**Envy Graph:**
```
    A       B       C
    (no edges - no one envies anyone!)
```

**Unenvied agents**: Alice, Bob, Charlie

**Action**: Give Item 4 to Alice (min index = 0)

**Updated Bundles:**
- Alice: {1, 4} → value = 8 + 2 = 10
- Bob: {2} → value = 9
- Charlie: {3} → value = 10

---

### Round 5: Allocating Item 5

**Item 5 values**: Alice=9, Bob=3, Charlie=2

**Bundle values:**
| Agent   | Values A's {1,4} | Values B's {2} | Values C's {3} |
|---------|------------------|----------------|----------------|
| Alice   | 10               | 3              | 7              |
| Bob     | 13               | 9              | 2              |
| Charlie | 8                | 6              | 10             |

**Envy Analysis:**
- Alice: 10 ≥ 3, 10 ≥ 7 → No envy
- Bob: 9 < 13 → **Bob envies Alice**
- Charlie: 10 ≥ 8, 10 ≥ 6 → No envy

**Envy Graph:**
```
    A ←─── B       C
```

**Unenvied agents**: Bob, Charlie (Alice is envied)

**Action**: Give Item 5 to Bob (min index among unenvied = 1)

**Updated Bundles:**
- Alice: {1, 4} → value = 10
- Bob: {2, 5} → value = 9 + 3 = 12
- Charlie: {3} → value = 10

---

### Round 6: Allocating Item 6

**Item 6 values**: Alice=4, Bob=7, Charlie=8

**Bundle values:**
| Agent   | Values A's {1,4} | Values B's {2,5} | Values C's {3} |
|---------|------------------|------------------|----------------|
| Alice   | 10               | 12               | 7              |
| Bob     | 13               | 12               | 2              |
| Charlie | 8                | 8                | 10             |

**Envy Analysis:**
- Alice: 10 < 12 → **Alice envies Bob**
- Bob: 12 < 13 → **Bob envies Alice**
- Charlie: 10 ≥ 8, 10 ≥ 8 → No envy

**Envy Graph:**
```
    A ←───→ B       C
    (mutual envy between A and B)
```

**Unenvied agents**: Charlie (only one not envied)

**Action**: Give Item 6 to Charlie

**Updated Bundles:**
- Alice: {1, 4} → value = 10
- Bob: {2, 5} → value = 12
- Charlie: {3, 6} → value = 10 + 8 = 18

---

### Round 7: Allocating Item 7 ⚠️ CYCLE ELIMINATION

**Item 7 values**: Alice=6, Bob=1, Charlie=9

**Bundle values:**
| Agent   | Values A's {1,4} | Values B's {2,5} | Values C's {3,6} |
|---------|------------------|------------------|------------------|
| Alice   | 10               | 12               | 11               |
| Bob     | 13               | 12               | 9                |
| Charlie | 8                | 8                | 18               |

**Envy Analysis:**
- Alice: 10 < 12 (envies B), 10 < 11 (envies C) → **Alice envies Bob and Charlie**
- Bob: 12 < 13 → **Bob envies Alice**
- Charlie: 18 ≥ 8, 18 ≥ 8 → No envy

**Envy Graph:**
```
    A ──→ B
    ↑     │
    └─────┘
    A also ──→ C
```

**Unenvied agents**: NONE! (Alice envied by Bob, Bob envied by Alice, Charlie envied by Alice)

---

### ⚡ CYCLE DETECTED: A → B → A

We have a cycle between Alice and Bob. We must eliminate it by rotating bundles.

**Cycle**: Alice → Bob → Alice

**Bundle Rotation:**
- Alice gets Bob's bundle {2, 5} (what Alice envies)
- Bob gets Alice's bundle {1, 4} (what Bob envies)

**After Rotation:**
| Agent   | Old Bundle | New Bundle | New Value |
|---------|------------|------------|-----------|
| Alice   | {1, 4}     | {2, 5}     | 3 + 9 = 12 |
| Bob     | {2, 5}     | {1, 4}     | 5 + 8 = 13 |
| Charlie | {3, 6}     | {3, 6}     | 18 (unchanged) |

**New Bundle Values After Rotation:**
| Agent   | Values A's {2,5} | Values B's {1,4} | Values C's {3,6} |
|---------|------------------|------------------|------------------|
| Alice   | 12               | 10               | 11               |
| Bob     | 12               | 13               | 9                |
| Charlie | 8                | 8                | 18               |

**New Envy Analysis:**
- Alice: 12 ≥ 10, 12 ≥ 11 → No envy
- Bob: 13 ≥ 12, 13 ≥ 9 → No envy
- Charlie: 18 ≥ 8, 18 ≥ 8 → No envy

**Unenvied agents**: Alice, Bob, Charlie (cycle eliminated!)

**Action**: Give Item 7 to Alice (min index = 0)

**Updated Bundles:**
- Alice: {2, 5, 7} → value = 12 + 6 = 18
- Bob: {1, 4} → value = 13
- Charlie: {3, 6} → value = 18

---

### Round 8: Allocating Item 8

**Item 8 values**: Alice=1, Bob=6, Charlie=4

**Bundle values:**
| Agent   | Values A's {2,5,7} | Values B's {1,4} | Values C's {3,6} |
|---------|--------------------| -----------------|------------------|
| Alice   | 18                 | 10               | 11               |
| Bob     | 13                 | 13               | 9                |
| Charlie | 17                 | 8                | 18               |

**Envy Analysis:**
- Alice: 18 ≥ 10, 18 ≥ 11 → No envy
- Bob: 13 ≥ 13, 13 ≥ 9 → No envy
- Charlie: 18 ≥ 17, 18 ≥ 8 → No envy

**Envy Graph:**
```
    A       B       C
    (no edges)
```

**Unenvied agents**: Alice, Bob, Charlie

**Action**: Give Item 8 to Alice (min index = 0)

**Updated Bundles:**
- Alice: {2, 5, 7, 8} → value = 18 + 1 = 19
- Bob: {1, 4} → value = 13
- Charlie: {3, 6} → value = 18

---

### Round 9: Allocating Item 9

**Item 9 values**: Alice=5, Bob=10, Charlie=1

**Bundle values:**
| Agent   | Values A's {2,5,7,8} | Values B's {1,4} | Values C's {3,6} |
|---------|----------------------|------------------|------------------|
| Alice   | 19                   | 10               | 11               |
| Bob     | 19                   | 13               | 9                |
| Charlie | 18                   | 8                | 18               |

**Envy Analysis:**
- Alice: 19 ≥ 10, 19 ≥ 11 → No envy
- Bob: 13 < 19 → **Bob envies Alice**
- Charlie: 18 ≥ 18, 18 ≥ 8 → No envy

**Envy Graph:**
```
    A ←─── B       C
```

**Unenvied agents**: Bob, Charlie

**Action**: Give Item 9 to Bob (min index among unenvied = 1)

**Updated Bundles:**
- Alice: {2, 5, 7, 8} → value = 19
- Bob: {1, 4, 9} → value = 13 + 10 = 23
- Charlie: {3, 6} → value = 18

---

### Round 10: Allocating Item 10

**Item 10 values**: Alice=10, Bob=4, Charlie=7

**Bundle values:**
| Agent   | Values A's {2,5,7,8} | Values B's {1,4,9} | Values C's {3,6} |
|---------|----------------------|--------------------|------------------|
| Alice   | 19                   | 15                 | 11               |
| Bob     | 19                   | 23                 | 9                |
| Charlie | 21                   | 9                  | 18               |

**Envy Analysis:**
- Alice: 19 ≥ 15, 19 ≥ 11 → No envy
- Bob: 23 ≥ 19, 23 ≥ 9 → No envy
- Charlie: 18 < 21 → **Charlie envies Alice**

**Envy Graph:**
```
    A ←─── C       B
```

**Unenvied agents**: Bob, Charlie (Alice is envied)

**Action**: Give Item 10 to Bob (min index among unenvied = 1)

**Updated Bundles:**
- Alice: {2, 5, 7, 8} → value = 19
- Bob: {1, 4, 9, 10} → value = 23 + 4 = 27
- Charlie: {3, 6} → value = 18

---

## Final Allocation

| Agent   | Bundle         | Items Received | Total Value (to self) |
|---------|----------------|----------------|-----------------------|
| Alice   | {2, 5, 7, 8}   | 4 items        | 3 + 9 + 6 + 1 = 19    |
| Bob     | {1, 4, 9, 10}  | 4 items        | 5 + 8 + 10 + 4 = 27   |
| Charlie | {3, 6}         | 2 items        | 10 + 8 = 18           |

---

## EF1 Verification

For an allocation to be **EF1**, any envy must be eliminable by removing one item from the envied bundle.

### Final Envy Analysis

| Agent   | Values A's bundle | Values B's bundle | Values C's bundle | Own Value |
|---------|-------------------|-------------------|-------------------|-----------|
| Alice   | 19                | 8+2+5+10=25       | 7+4=11            | 19        |
| Bob     | 9+3+1+6=19        | 27                | 2+7=9             | 27        |
| Charlie | 6+2+9+4=21        | 3+5+1+7=16        | 18                | 18        |

**Checking each agent:**

### Alice
- Values own bundle: 19
- Values Bob's bundle: 25 → **Alice envies Bob!**
- Values Charlie's bundle: 11 → No envy

**Is Alice's envy EF1?** Remove items from Bob's bundle {1, 4, 9, 10}:
- Remove Item 1: Alice values {4,9,10} = 2+5+10 = 17 < 19 → **No envy!**

**Removing Item 1 eliminates Alice's envy.** ✓

### Bob
- Values own bundle: 27
- Values Alice's bundle: 19 → No envy
- Values Charlie's bundle: 9 → No envy
- **Status: Envy-free** ✓

### Charlie
- Values own bundle: 18
- Values Alice's bundle: 21 → **Charlie envies Alice!**
- Values Bob's bundle: 16 → No envy

**Is Charlie's envy EF1?** Remove items from Alice's bundle {2, 5, 7, 8}:
- Remove Item 2: Charlie values {5,7,8} = 2+9+4 = 15 < 18 → **No envy!**

**Removing Item 2 eliminates Charlie's envy.** ✓

### Conclusion: The allocation is EF1 ✓

---

## Key Observations

### 1. Cycle Elimination in Action

Unlike many examples, this walkthrough includes **actual cycle elimination** at Round 7. When Alice and Bob had mutual envy (A→B and B→A), we rotated their bundles. This is the key mechanism that guarantees ECE always finds an unenvied agent.

### 2. Why ECE Guarantees EF1

When an agent receives an item, they are **unenvied** at that moment. Any future envy toward that agent can be traced to items received after that point. Removing any one of those items eliminates the envy.

### 3. Bundle Swapping Preserves Happiness

After the cycle elimination in Round 7:
- Alice went from valuing her bundle at 10 to valuing it at 12 (she got what she envied)
- Bob went from valuing his bundle at 12 to valuing it at 13 (he got what he envied)

No one is worse off after a cycle swap!

### 4. Tie-Breaking Matters

The deterministic tie-breaking rule (smallest agent index) means:
- Alice (index 0) gets priority when all are unenvied
- This leads to a different allocation than arbitrary tie-breaking would

### 5. Unequal Bundle Sizes

Alice and Bob got 4 items each, while Charlie got only 2. ECE prioritizes **fairness (EF1)** over **equality**. Charlie received fewer items because the items she got (3 and 6) were highly valued by her.

### 6. Comparison to Neural Network Approach

The **FATransformer** in this repository takes a different approach:

| Aspect | ECE | FATransformer |
|--------|-----|---------------|
| Fairness guarantee | EF1 (always) | Often EF1, not guaranteed |
| Welfare optimization | No explicit optimization | Maximizes Nash Welfare |
| Computational cost | O(nm²) | O(forward pass) |
| Allocation type | Discrete | Probabilistic → Discrete |

---

## Verifying This Example

You can verify this example matches the implementation:

```python
import numpy as np
import sys
sys.path.insert(0, 'eval_pipeline')
from utils.inference import get_ece_allocation

valuation_matrix = np.array([
    [8,  3,  7,  2,  9,  4,  6,  1,  5, 10],  # Alice
    [5,  9,  2,  8,  3,  7,  1,  6, 10,  4],  # Bob
    [3,  6, 10,  5,  2,  8,  9,  4,  1,  7],  # Charlie
], dtype=float)

allocation = get_ece_allocation(valuation_matrix)
print("Alice:", [i+1 for i in range(10) if allocation[0][i] == 1])  # [2, 5, 7, 8]
print("Bob:", [i+1 for i in range(10) if allocation[1][i] == 1])    # [1, 4, 9, 10]
print("Charlie:", [i+1 for i in range(10) if allocation[2][i] == 1]) # [3, 6]
```

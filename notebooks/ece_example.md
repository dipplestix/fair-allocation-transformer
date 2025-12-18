# Envy-Cycle Elimination (ECE) Algorithm: A Detailed Example

This document provides a step-by-step walkthrough of the Envy-Cycle Elimination algorithm for fair division, demonstrating how it achieves an EF1 (envy-free up to one item) allocation.

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

    4. Give item g to the unenvied agent
```

### Why Cycles Must Be Eliminated

If there's a cycle (e.g., A→B→C→A), then every agent is envied by someone, so no agent is "unenvied." By rotating bundles along the cycle, we break the cycle without making anyone worse off (each agent gets a bundle they value at least as much).

---

## Example Setup

### Agents
- **Alice (A)**: Prefers items 10, 5, 1
- **Bob (B)**: Prefers items 9, 2, 4
- **Charlie (C)**: Prefers items 3, 7, 6

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
- Alice's bundle: {} → v_A = 0
- Bob's bundle: {} → v_B = 0
- Charlie's bundle: {} → v_C = 0

---

### Round 1: Allocating Item 1

**Item 1 values**: Alice=8, Bob=5, Charlie=3

**Current bundle values:**
| Agent   | Values Alice's {} | Values Bob's {} | Values Charlie's {} | Own Bundle Value |
|---------|-------------------|-----------------|---------------------|------------------|
| Alice   | 0                 | 0               | 0                   | 0                |
| Bob     | 0                 | 0               | 0                   | 0                |
| Charlie | 0                 | 0               | 0                   | 0                |

**Envy Graph:**
```
    A       B       C
    (no edges - no one envies anyone with empty bundles)
```

**Unenvied agents**: All agents (A, B, C)

**Action**: Give Item 1 to Alice (choosing arbitrarily among unenvied agents)

**Updated Bundles:**
- Alice: {1} → v_A({1}) = 8
- Bob: {} → v_B({}) = 0
- Charlie: {} → v_C({}) = 0

---

### Round 2: Allocating Item 2

**Item 2 values**: Alice=3, Bob=9, Charlie=6

**Current bundle values (from each agent's perspective):**
| Agent   | Values Alice's {1} | Values Bob's {} | Values Charlie's {} | Own Value |
|---------|--------------------|-----------------|---------------------|-----------|
| Alice   | 8                  | 0               | 0                   | 8         |
| Bob     | 5                  | 0               | 0                   | 0         |
| Charlie | 3                  | 0               | 0                   | 0         |

**Envy Analysis:**
- Alice: 8 ≥ 0, 8 ≥ 0 → No envy
- Bob: 0 < 5 → Bob envies Alice
- Charlie: 0 < 3 → Charlie envies Alice

**Envy Graph:**
```
        ┌───────┐
        │   A   │ ←─── B
        └───────┘
            ↑
            │
            C
```

**Unenvied agents**: Bob, Charlie (Alice is envied)

**Action**: Give Item 2 to Bob (choosing arbitrarily)

**Updated Bundles:**
- Alice: {1} → v_A = 8
- Bob: {2} → v_B = 9
- Charlie: {} → v_C = 0

---

### Round 3: Allocating Item 3

**Item 3 values**: Alice=7, Bob=2, Charlie=10

**Current bundle values:**
| Agent   | Values A's {1} | Values B's {2} | Values C's {} | Own Value |
|---------|----------------|----------------|---------------|-----------|
| Alice   | 8              | 3              | 0             | 8         |
| Bob     | 5              | 9              | 0             | 9         |
| Charlie | 3              | 6              | 0             | 0         |

**Envy Analysis:**
- Alice: 8 ≥ 3, 8 ≥ 0 → No envy
- Bob: 9 ≥ 5, 9 ≥ 0 → No envy
- Charlie: 0 < 3 (envies A), 0 < 6 (envies B)

**Envy Graph:**
```
    A ←─── C ───→ B
```

**Unenvied agents**: Alice, Bob

**Action**: Give Item 3 to Alice

**Updated Bundles:**
- Alice: {1, 3} → v_A = 8 + 7 = 15
- Bob: {2} → v_B = 9
- Charlie: {} → v_C = 0

---

### Round 4: Allocating Item 4

**Item 4 values**: Alice=2, Bob=8, Charlie=5

**Current bundle values:**
| Agent   | Values A's {1,3} | Values B's {2} | Values C's {} | Own Value |
|---------|------------------|----------------|---------------|-----------|
| Alice   | 8+7=15           | 3              | 0             | 15        |
| Bob     | 5+2=7            | 9              | 0             | 9         |
| Charlie | 3+10=13          | 6              | 0             | 0         |

**Envy Analysis:**
- Alice: 15 ≥ 3, 15 ≥ 0 → No envy
- Bob: 9 ≥ 7, 9 ≥ 0 → No envy
- Charlie: 0 < 13 (envies A), 0 < 6 (envies B)

**Envy Graph:**
```
    A ←─── C ───→ B
```

**Unenvied agents**: Alice, Bob

**Action**: Give Item 4 to Bob

**Updated Bundles:**
- Alice: {1, 3} → v_A = 15
- Bob: {2, 4} → v_B = 9 + 8 = 17
- Charlie: {} → v_C = 0

---

### Round 5: Allocating Item 5

**Item 5 values**: Alice=9, Bob=3, Charlie=2

**Current bundle values:**
| Agent   | Values A's {1,3} | Values B's {2,4} | Values C's {} | Own Value |
|---------|------------------|------------------|---------------|-----------|
| Alice   | 15               | 3+2=5            | 0             | 15        |
| Bob     | 7                | 17               | 0             | 17        |
| Charlie | 13               | 6+5=11           | 0             | 0         |

**Envy Analysis:**
- Alice: No envy (15 ≥ 5, 15 ≥ 0)
- Bob: No envy (17 ≥ 7, 17 ≥ 0)
- Charlie: Envies A (0 < 13), Envies B (0 < 11)

**Envy Graph:**
```
    A ←─── C ───→ B
```

**Unenvied agents**: Alice, Bob

**Action**: Give Item 5 to Alice

**Updated Bundles:**
- Alice: {1, 3, 5} → v_A = 15 + 9 = 24
- Bob: {2, 4} → v_B = 17
- Charlie: {} → v_C = 0

---

### Round 6: Allocating Item 6

**Item 6 values**: Alice=4, Bob=7, Charlie=8

**Current bundle values:**
| Agent   | Values A's {1,3,5} | Values B's {2,4} | Values C's {} | Own Value |
|---------|--------------------|------------------|---------------|-----------|
| Alice   | 8+7+9=24           | 3+2=5            | 0             | 24        |
| Bob     | 5+2+3=10           | 9+8=17           | 0             | 17        |
| Charlie | 3+10+2=15          | 6+5=11           | 0             | 0         |

**Envy Analysis:**
- Alice: No envy
- Bob: No envy
- Charlie: Envies A (0 < 15), Envies B (0 < 11)

**Envy Graph:**
```
    A ←─── C ───→ B
```

**Unenvied agents**: Alice, Bob

**Action**: Give Item 6 to Charlie (finally!)

**Updated Bundles:**
- Alice: {1, 3, 5} → v_A = 24
- Bob: {2, 4} → v_B = 17
- Charlie: {6} → v_C = 8

---

### Round 7: Allocating Item 7

**Item 7 values**: Alice=6, Bob=1, Charlie=9

**Current bundle values:**
| Agent   | Values A's {1,3,5} | Values B's {2,4} | Values C's {6} | Own Value |
|---------|--------------------|------------------|----------------|-----------|
| Alice   | 24                 | 5                | 4              | 24        |
| Bob     | 10                 | 17               | 7              | 17        |
| Charlie | 15                 | 11               | 8              | 8         |

**Envy Analysis:**
- Alice: No envy (24 ≥ 5, 24 ≥ 4)
- Bob: No envy (17 ≥ 10, 17 ≥ 7)
- Charlie: 8 < 15 (envies A), 8 < 11 (envies B)

**Envy Graph:**
```
    A ←─── C ───→ B
```

**Unenvied agents**: Alice, Bob

**Action**: Give Item 7 to Charlie

**Updated Bundles:**
- Alice: {1, 3, 5} → v_A = 24
- Bob: {2, 4} → v_B = 17
- Charlie: {6, 7} → v_C = 8 + 9 = 17

---

### Round 8: Allocating Item 8

**Item 8 values**: Alice=1, Bob=6, Charlie=4

**Current bundle values:**
| Agent   | Values A's {1,3,5} | Values B's {2,4} | Values C's {6,7} | Own Value |
|---------|--------------------|------------------|------------------|-----------|
| Alice   | 24                 | 5                | 4+6=10           | 24        |
| Bob     | 10                 | 17               | 7+1=8            | 17        |
| Charlie | 15                 | 11               | 8+9=17           | 17        |

**Envy Analysis:**
- Alice: No envy
- Bob: No envy
- Charlie: 17 ≥ 15? Yes! 17 ≥ 11? Yes! → **No envy!**

**Envy Graph:**
```
    A       B       C
    (no edges!)
```

**Unenvied agents**: All (A, B, C)

**Action**: Give Item 8 to Bob

**Updated Bundles:**
- Alice: {1, 3, 5} → v_A = 24
- Bob: {2, 4, 8} → v_B = 17 + 6 = 23
- Charlie: {6, 7} → v_C = 17

---

### Round 9: Allocating Item 9

**Item 9 values**: Alice=5, Bob=10, Charlie=1

**Current bundle values:**
| Agent   | Values A's {1,3,5} | Values B's {2,4,8} | Values C's {6,7} | Own Value |
|---------|--------------------|--------------------| -----------------|-----------|
| Alice   | 24                 | 5+6=11             | 10               | 24        |
| Bob     | 10                 | 23                 | 8                | 23        |
| Charlie | 15                 | 11+4=15            | 17               | 17        |

**Envy Analysis:**
- Alice: No envy (24 ≥ 11, 24 ≥ 10)
- Bob: No envy (23 ≥ 10, 23 ≥ 8)
- Charlie: 17 ≥ 15? Yes! 17 ≥ 15? Yes! → No envy

**Envy Graph:**
```
    A       B       C
    (no edges!)
```

**Unenvied agents**: All (A, B, C)

**Action**: Give Item 9 to Bob (he values it most)

**Updated Bundles:**
- Alice: {1, 3, 5} → v_A = 24
- Bob: {2, 4, 8, 9} → v_B = 23 + 10 = 33
- Charlie: {6, 7} → v_C = 17

---

### Round 10: Allocating Item 10

**Item 10 values**: Alice=10, Bob=4, Charlie=7

**Current bundle values:**
| Agent   | Values A's {1,3,5} | Values B's {2,4,8,9} | Values C's {6,7} | Own Value |
|---------|--------------------|----------------------|------------------|-----------|
| Alice   | 24                 | 5+6+5=16             | 10               | 24        |
| Bob     | 10                 | 33                   | 8                | 33        |
| Charlie | 15                 | 15+4+1=20            | 17               | 17        |

**Envy Analysis:**
- Alice: No envy (24 ≥ 16, 24 ≥ 10)
- Bob: No envy (33 ≥ 10, 33 ≥ 8)
- Charlie: 17 ≥ 15? Yes! But 17 < 20 → **Charlie envies Bob!**

**Envy Graph:**
```
    A       B ←─── C
```

**Unenvied agents**: Alice, Charlie

**Action**: Give Item 10 to Alice

**Updated Bundles:**
- Alice: {1, 3, 5, 10} → v_A = 24 + 10 = 34
- Bob: {2, 4, 8, 9} → v_B = 33
- Charlie: {6, 7} → v_C = 17

---

## Final Allocation

| Agent   | Bundle         | Items Received | Total Value (to self) |
|---------|----------------|----------------|-----------------------|
| Alice   | {1, 3, 5, 10}  | 4 items        | 8 + 7 + 9 + 10 = 34   |
| Bob     | {2, 4, 8, 9}   | 4 items        | 9 + 8 + 6 + 10 = 33   |
| Charlie | {6, 7}         | 2 items        | 8 + 9 = 17            |

---

## EF1 Verification

For an allocation to be **EF1**, any envy must be eliminable by removing one item from the envied bundle.

### Final Envy Analysis

| Agent   | Values A's bundle | Values B's bundle | Values C's bundle | Own Value |
|---------|-------------------|-------------------|-------------------|-----------|
| Alice   | 34                | 3+2+1+5=11        | 4+6=10            | 34        |
| Bob     | 5+2+3+4=14        | 33                | 7+1=8             | 33        |
| Charlie | 3+10+2+7=22       | 6+5+4+1=16        | 17                | 17        |

**Checking each agent:**

### Alice
- Values own bundle: 34
- Values Bob's bundle: 11 → No envy
- Values Charlie's bundle: 10 → No envy
- **Status: Envy-free**

### Bob
- Values own bundle: 33
- Values Alice's bundle: 14 → No envy
- Values Charlie's bundle: 8 → No envy
- **Status: Envy-free**

### Charlie
- Values own bundle: 17
- Values Alice's bundle: 22 → **Envies Alice!**
- Values Bob's bundle: 16 → No envy

**Charlie envies Alice.** Is this EF1?

Remove each item from Alice's bundle and check:
- Remove Item 1: Charlie values {3,5,10} = 10+2+7 = 19 > 17 → Still envies
- Remove Item 3: Charlie values {1,5,10} = 3+2+7 = 12 < 17 → **No envy!**
- Remove Item 5: Charlie values {1,3,10} = 3+10+7 = 20 > 17 → Still envies
- Remove Item 10: Charlie values {1,3,5} = 3+10+2 = 15 < 17 → **No envy!**

**Removing Item 3 or Item 10 eliminates Charlie's envy.**

### Conclusion: The allocation is EF1 ✓

---

## Key Observations

### 1. Why ECE Guarantees EF1

The key insight is that when an agent receives an item, they are **unenvied** at that moment. Any future envy toward that agent can be traced to the new items they receive. Removing the most recently received item (or any single high-value item) eliminates the envy.

### 2. No Cycle Elimination Needed in This Example

Interestingly, this particular run of ECE didn't require any cycle elimination — there was always at least one unenvied agent. This happens when preferences are sufficiently diverse. In cases with more similar preferences, cycles occur more frequently.

### 3. Unequal Bundle Sizes

Notice that Alice and Bob got 4 items each while Charlie only got 2. ECE doesn't guarantee equal numbers of items — it prioritizes fairness (EF1) over equality. Charlie got fewer items because the items they received were highly valued by them.

### 4. Comparison to Neural Network Approach

The **FATransformer** in this repository takes a different approach:
- ECE processes items sequentially with guaranteed EF1
- FATransformer maximizes **Nash Welfare** (geometric mean of utilities)
- FATransformer produces probabilistic allocations that can then be rounded

Both approaches have trade-offs:
| Aspect | ECE | FATransformer |
|--------|-----|---------------|
| Fairness guarantee | EF1 (always) | Often EF1, not guaranteed |
| Welfare optimization | No explicit optimization | Maximizes Nash Welfare |
| Computational cost | O(nm²) | O(forward pass) |
| Allocation type | Discrete | Probabilistic → Discrete |

---

## Appendix: What If We Had a Cycle?

To illustrate cycle elimination, consider a hypothetical scenario with these bundle values:

| Agent | Values A's bundle | Values B's bundle | Values C's bundle |
|-------|-------------------|-------------------|-------------------|
| A     | 10                | 15                | 8                 |
| B     | 12                | 10                | 18                |
| C     | 20                | 9                 | 15                |

**Envy Graph:**
```
    ┌─────────────────────────────┐
    │                             │
    ▼                             │
    A ──────→ B ──────→ C ────────┘
```

This forms a cycle: A → B → C → A

**Cycle Elimination:**
1. Identify cycle: A → B → C → A
2. Rotate bundles along the cycle:
   - A gets B's bundle (what A envies)
   - B gets C's bundle (what B envies)
   - C gets A's bundle (what C envies)

After rotation:
- Everyone is at least as happy (they get what they envied)
- The cycle is broken
- We can now find an unenvied agent

This mechanism ensures we can always make progress toward a fair allocation.

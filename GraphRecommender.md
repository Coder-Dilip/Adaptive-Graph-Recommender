

# The Hybrid Graph Recommender: Algorithm Documentation

## 1\. High-Level Overview

This system is a **Hybrid Recommender**. Most systems are either:

  * **Content-Based:** "You liked an apple, here is a pear." (Similarity)
  * **Collaborative/Behavioral:** "People who bought apples also bought chargers." (User Behavior)

**Our algorithm combines both.** It starts with a "Cold Start" knowledge base derived from AI text analysis (Content-Based) and evolves into a "Behavioral Brain" as users interact with it (Reinforcement Learning).

-----

## 2\. The Architecture

### The Core Components

1.  **The Semantic Brain (BERT):** A pre-trained AI model (`all-MiniLM-L6-v2`) that turns text into numbers (vectors). It understands that "Burger" and "Sandwich" are related, even if the words look different.
2.  **The Graph (NetworkX):** A spiderweb of connections.
      * **Nodes:** The recipes (e.g., "Chicken Burger").
      * **Edges:** The relationship between them. The thicker the line (weight), the stronger the connection.
3.  **The Memory (Pickle + SQLite):**
      * **SQLite:** Stores the undeniable facts (Recipe names, descriptions, total clicks).
      * **Pickle:** Stores the "learned intuition" (The Graph structure and edge weights).

-----

## 3\. The Workflow: Step-by-Step

### Phase 1: The "Cold Start" (Initialization)

*Problem:* When the app first launches, it has 0 users and 0 clicks. How does it know what to recommend?
*Solution:* **Semantic Initialization.**

1.  The AI reads every recipe description.
2.  It calculates **Cosine Similarity** (how close are their meanings?).
3.  It draws the initial graph edges. If "Veg Burger" and "Veg Salad" are 80% similar, it draws a line with `weight=0.8`.
4.  **Result:** The system is "born smart."

-----

### Phase 2: The Search Process (The "Read" Path)

When a user searches for "Spicy Food", the system performs a 3-step calculation:

#### Step A: Vector Search (Finding Candidates)

The system converts "Spicy Food" into a vector and finds the top 5 recipes that mathematically match that concept (e.g., "Pepperoni Pizza", "Arrabiata Pasta").

#### Step B: Hybrid Re-Ranking (The "Wisdom" Layer)

The system doesn't just show the most relevant items. It asks: *"Is this relevant item actually good?"*
It calculates a **Hybrid Score** for each candidate:

$$Score = (\alpha \times \text{Similarity}) + (\beta \times \text{Popularity})$$

  * **$\alpha$ (Alpha 0.6):** How much we care about the text match.
  * **$\beta$ (Beta 0.4):** How much we care about user popularity.
  * **Log Normalization:** We use `log(hits)` for popularity. This ensures a recipe with 1,000 clicks isn't unfairly ranked 1,000x higher than a recipe with 10 clicks. It balances the playing field.

#### Step C: The Probabilistic Graph Walk (Discovery)

Once we pick the "Winner" from Step B (The Start Node), we don't just stop. We walk the graph to find related items.

  * **Why?** If you search "Burger", the Start Node is "Chicken Burger". But you might want "Fries" (which isn't semantically similar to "Burger" but is *behaviorally* linked).
  * The Walker moves 4 steps. At every step, it flips a coin:
      * **75% Chance (Exploit):** Go to the strongest connection.
      * **25% Chance (Explore):** Try a random connection. This enables **Serendipity** (happy accidents).

-----

### Phase 3: The Learning Process (The "Write" Path)

This is where the AI gets smarter. When a user **CLICKS** a result, the `/track-click` endpoint triggers.

#### Step A: Update Popularity (Global Score)

The `hits` counter for that item increases by +1. This item will now rank slightly higher in future Hybrid Scores.

#### Step B: Global Decay (The "Forgetting" Mechanism)

Before learning the new thing, the system slightly weakens **every single connection in the graph** by multiplying weights by `0.995`.

  * **Why?** Trends change. "Pumpkin Spice Latte" is popular in October. If we don't let weights decay, it would still be recommended in July. Decay allows the system to "forget" old trends.

#### Step C: Associative Learning (Connecting the Dots)

The system looks at the User's Intent (`Query`) and the Result (`Clicked Item`).

  * **Scenario:** User searched "Healthy" but clicked "Cheeseburger".
  * **Action:** The system draws (or strengthens) a line between the "Healthy" concept node and the "Cheeseburger" node.
  * **The Math:** It uses an **Asymptotic Update**:
    $$W_{new} = W_{old} + 0.1 \times (5.0 - W_{old})$$
    This ensures the weight grows but never explodes to infinity. It naturally caps at 5.0.

-----

## 4\. Technical Safety Features

### Atomic Persistence (Data Safety)

Saving a complex graph to a file takes time (milliseconds to seconds). If the server crashes halfway through saving, the file gets corrupted and the brain is lost.
**Our Solution:**

1.  Write the graph to `graph_state.pkl.tmp` (Temporary file).
2.  Once writing is 100% complete, perform an **OS Atomic Swap** to rename it to `graph_state.pkl`.
3.  This guarantees the file is always valid.

### Thread Locking (Concurrency)

In a web server, multiple users click at the same time.
**The Risk:** User A reads `Hits=10`. User B reads `Hits=10`. User A writes `11`. User B writes `11`. We lost a click.
**The Solution:** We use `threading.Lock()`.

```python
with self._lock:
    # Only one thread can enter this block at a time
    self.G.nodes[item]['hits'] += 1
```

This forces users to form a single-file line when updating the brain, ensuring 100% accuracy.

-----

## 5\. Glossary of Terms

| Term | Definition |
| :--- | :--- |
| **Embedding** | Converting text into a list of numbers (vector) so a computer can understand its meaning. |
| **Cosine Similarity** | A math formula to measure how similar two vectors are (1.0 = Identical, 0.0 = Unrelated). |
| **Node** | An item in the graph (e.g., A specific Recipe). |
| **Edge** | The connection line between two nodes. |
| **Weight** | The thickness of the edge. Represents the strength of the relationship. |
| **Epsilon-Greedy** | The strategy of mostly choosing the best option (Exploitation) but sometimes choosing a random option (Exploration) to discover new things. |
| **Race Condition** | A bug where two processes try to change data at the same time, causing errors. Solved by Locking. |
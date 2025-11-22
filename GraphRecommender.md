# 🧠 The RobustRecommender Engine: A Comprehensive Guide

## 1. Introduction

Welcome to the documentation for the RobustRecommender. This system is the "brain"
behind a research paper discovery platform. It helps users find relevant papers not just by
matching keywords, but by understanding human behavior and semantic meaning.

### What kind of Recommender is this?

It is a **Hybrid System**. It combines two powerful techniques:

1. **Content-Based Filtering (The "Librarian")**: It reads the text (Title + Summary) to  
   understand what a paper is about.

2. **Collaborative Filtering (The "Social Network")**: It watches what users click to  
   understand which papers are related, even if they don't sound alike.

---

## 2. The Core Architecture

The system is built on three pillars:

| Component            | Technology                                | Analogy                                                                 |
|----------------------|---------------------------------------------|-------------------------------------------------------------------------|
| The Semantic Engine  | FAISS (Facebook AI Similarity Search)       | A super-fast librarian who knows exactly where every book is located.  |
| The Behavior Graph   | NetworkX (Graph Database)                   | A spiderweb connecting papers together.                                 |
| The Neural Brain     | Sentence Transformers (BERT)                | A translator turning English text into numeric vectors.                 |

---

## 3. Key Concepts Explained

### A. Embeddings & Vectors

Computers can't read English. To them, *"Dog"* and *"Puppy"* are different words.  
To fix this, we use **Embeddings** — turning each paper into a vector of 384 numbers.

- **Concept:** Imagine a 3D map. "Dog" and "Puppy" are close. "Cat" is nearby. "Car" is far.  
- **Our Code:** We use **all-MiniLM-L6-v2** to generate vectors.

### B. The Graph (Nodes & Edges)

- **Node:** A single research paper  
- **Edge:** A connection between two papers  
- **Weight:** How strong the link is (e.g., 100 clicks = thick highway)

### C. "Time-Gravity" (The Decay Logic)

We don’t let old papers dominate forever.

- We check when each link was last clicked.
- Old edges decay unless refreshed by user clicks.

---

## 4. How It Works: The Four Main Functions

### ① Initialization (`__init__`)

What it does: **Wakes up the brain**.

1. Loads the neural model  
2. Reads all papers from the DB  
3. Builds FAISS index  
4. Loads graph from disk (.pkl)

---

### ② Search (`search`)

The problem: A keyword search may miss semantic matches.  
A graph-only search misses new papers.

**Solution: Merge Strategy**

1. **Semantic Source (FAISS):** Finds Top-50 meaning-based papers  
2. **Graph Source:** Picks Top-20 behaviorally related papers  
3. **Merge:** Combine & rank  
4. **Rescue Boost:** Graph items get a bonus score

---

### ③ Similar Items (`get_similar_items`)

Goal: Suggest *Next Best Paper*.

Method: **Beam Search**

1. Create frontier pool of neighbors  
2. Score = `(Weight + Popularity) * Semantic_Match`  
3. **70/30 Rule:**  
   - 70% deterministic  
   - 30% random among Top 5  
4. Expand pool with decay penalty (×0.6)

---

### ④ Learning (`process_user_click`)

When user clicks:

1. Increase paper popularity  
2. Strengthen graph link (+0.5)  
3. Update timestamp  
4. Lazy decay (done later to avoid blocking)

---

## 5. Scalability: Why It Doesn’t Crash

| Feature           | Why It Matters |
|-------------------|----------------|
| **FAISS Integration** | Search becomes O(log N), enabling millions scale |
| **Thread Locks** | Prevent graph corruption when many users click simultaneously |
| **Lazy Decay** | Eliminates huge recomputation loops |
| **Heap Selection** | Fast trending search using O(N log K) |

---

## 6. Summary for the Developer

- **Input:** List of paper dictionaries  
- **Storage:** `.pkl` graph + FAISS index  
- **Output:** JSON recommendations  
- **Maintenance:** None — Time-Gravity auto-cleans the system  
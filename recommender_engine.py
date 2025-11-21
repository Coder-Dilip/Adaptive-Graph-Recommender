import networkx as nx
import numpy as np
import pickle
import os
import math
import random
import threading  # <--- Added for thread safety
import shutil     # <--- Added for atomic file moves
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RobustRecommender:
    def __init__(self, recipes_data, model_name='all-MiniLM-L6-v2', storage_file='graph_state.pkl'):
        """
        recipes_data: List of dicts from DB.
        storage_file: .pkl file to store the graph.
        """
        self.recipes = recipes_data
        self.recipe_map = {r['name']: r for r in recipes_data}
        self.recipe_names = [r['name'] for r in recipes_data]
        self.storage_file = storage_file
        
        # --- THREAD LOCK ---
        # This lock ensures that only one thread can update or save the graph at a time.
        self._lock = threading.Lock() 
        
        # Hyperparameters
        self.ALPHA = 0.6        # Importance of Semantic Similarity
        self.BETA = 0.4         # Importance of Popularity
        self.LEARNING_RATE = 0.1
        self.DECAY_FACTOR = 0.995
        self.MAX_WEIGHT = 5.0
        
        # Load Model
        print("⏳ (Recommender) Loading Transformer Model...")
        self.model = SentenceTransformer(model_name)
        
        # Compute embeddings for all items once
        self.item_embeddings = self.model.encode([r['name'] + " " + r['description'] for r in recipes_data])

        # Initialize Graph
        if os.path.exists(self.storage_file):
            self.load_graph()
            self._sync_popularity_from_db() # Ensure graph matches DB truth
        else:
            self.build_cold_start_graph()

    def build_cold_start_graph(self, k=4):
        print("🆕 (Recommender) Building Cold Start Graph...")
        self.G = nx.Graph()
        sim_matrix = cosine_similarity(self.item_embeddings)
        np.fill_diagonal(sim_matrix, 0)

        for i, name in enumerate(self.recipe_names):
            # Initialize hits from DB data
            db_hits = self.recipe_map[name].get('popularity', 0)
            self.G.add_node(name, hits=db_hits)
            
            # Top K connections
            neighbors_idx = sim_matrix[i].argsort()[-k:]
            for idx in neighbors_idx:
                weight = max(0.1, float(sim_matrix[i][idx]))
                self.G.add_edge(name, self.recipe_names[idx], weight=weight)

    def _sync_popularity_from_db(self):
        """Ensures RAM graph has the same popularity counts as the SQLite DB"""
        with self._lock:
            for name, data in self.recipe_map.items():
                if self.G.has_node(name):
                    self.G.nodes[name]['hits'] = data['popularity']

    def _calculate_hybrid_score(self, similarity, node_hits):
        """Score = Alpha*Sim + Beta*Log(Hits)"""
        popularity_score = math.log1p(node_hits) 
        # Cap popularity influence so it doesn't drown out relevance
        norm_pop = min(popularity_score / 5.0, 1.0) 
        return (self.ALPHA * similarity) + (self.BETA * norm_pop)

    def search(self, query, num_recs=3):
        """Returns ranked results based on Hybrid Score + Graph Walk"""
        # 1. Vector Search
        query_emb = self.model.encode([query])
        sims = cosine_similarity(query_emb, self.item_embeddings)[0]
        
        # 2. Hybrid Ranking (Relevance + Popularity)
        candidates = []
        # We strictly read values here, which is generally thread-safe in Python,
        # but if you want to be 100% paranoid, you could wrap this loop in a lock.
        # For performance, we usually allow reads to be lock-free in this context.
        for i, name in enumerate(self.recipe_names):
            hits = self.G.nodes[name].get('hits', 0)
            score = self._calculate_hybrid_score(sims[i], hits)
            candidates.append((name, score))
        
        # Sort by hybrid score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 3. Select Top Candidate and Walk Graph
        best_match = candidates[0][0]
        path = self._graph_walk(best_match, steps=num_recs)
        
        # Format output
        results = []
        for name in path:
            item = self.recipe_map[name].copy()
            item['current_hits'] = self.G.nodes[name].get('hits', 0)
            results.append(item)
            
        return {"start_node": best_match, "recommendations": results}

    def get_similar_items(self, item_name, num_recs=3):
        """Item-to-Item Recommendation (No Search Query)"""
        if item_name not in self.G.nodes:
            return []
        
        # Start walking directly from the item
        path = self._graph_walk(item_name, steps=num_recs)
        
        results = []
        for name in path:
            if name == item_name: continue # Skip self
            item = self.recipe_map[name].copy()
            item['current_hits'] = self.G.nodes[name].get('hits', 0)
            results.append(item)
        return results

    def get_trending(self, limit=5):
        """Returns top items purely by Hits (Popularity)"""
        # Sort nodes by 'hits' attribute
        sorted_nodes = sorted(self.G.nodes(data=True), key=lambda x: x[1].get('hits', 0), reverse=True)
        
        results = []
        for name, data in sorted_nodes[:limit]:
            item = self.recipe_map[name].copy()
            item['current_hits'] = data.get('hits', 0)
            results.append(item)
        return results

    def get_graph_data(self):
        """Exports JSON for frontend visualization libraries"""
        return nx.node_link_data(self.G)

    def _graph_walk(self, start_node, steps):
        """Probabilistic walk from start node"""
        current = start_node
        path = [current]
        visited = {current}
        
        # Fix: Ensure we get 'steps' amount of NEW recommendations
        target_length = steps + 1
        max_attempts = steps * 5 
        attempts = 0
        
        while len(path) < target_length and attempts < max_attempts:
            attempts += 1
            neighbors = list(self.G.neighbors(current))
            
            if not neighbors: break
            
            # Exploration Logic (25% chance)
            if random.random() < 0.25:
                weights = np.array([self.G[current][n]['weight'] for n in neighbors])
                probs = weights / weights.sum()
                next_node = np.random.choice(neighbors, p=probs)
            else:
                next_node = max(neighbors, key=lambda n: self.G[current][n]['weight'])
            
            if next_node not in visited:
                path.append(next_node)
                visited.add(next_node)
            
            # Always move, allowing traversal through visited nodes
            current = next_node
        
        return path

    def process_user_click(self, query, clicked_item):
        """
        Updates graph weights based on user action.
        ATOMIC OPERATION: Protected by Lock.
        """
        if clicked_item not in self.G.nodes: return

        # --- ENTER CRITICAL SECTION ---
        with self._lock:
            # 1. Update Hits (RAM)
            self.G.nodes[clicked_item]['hits'] += 1
            
            # 2. Decay all edges slightly
            for u, v, d in self.G.edges(data=True):
                d['weight'] *= self.DECAY_FACTOR
                
            # 3. Reinforce connection: Query Context -> Clicked Item
            query_emb = self.model.encode([query])
            sims = cosine_similarity(query_emb, self.item_embeddings)[0]
            start_node = self.recipe_names[np.argmax(sims)]
            
            # If they are different, link them!
            if start_node != clicked_item:
                if self.G.has_edge(start_node, clicked_item):
                    # Reinforce existing
                    old_w = self.G[start_node][clicked_item]['weight']
                    new_w = old_w + self.LEARNING_RATE * (self.MAX_WEIGHT - old_w)
                    self.G[start_node][clicked_item]['weight'] = new_w
                else:
                    # Create new "Shortcut" connection
                    self.G.add_edge(start_node, clicked_item, weight=0.5)
        # --- EXIT CRITICAL SECTION ---

    def save_to_disk(self):
        """
        Saves the graph object to pickle atomically.
        Prevents corrupted files if the process crashes mid-write.
        """
        # 1. Write to a temporary file first
        temp_file = self.storage_file + ".tmp"
        
        # Lock strictly during the write/copy phase
        with self._lock:
            with open(temp_file, 'wb') as f:
                pickle.dump(self.G, f)
        
        # 2. Atomic Swap (Rename temp to actual)
        # os.replace is atomic on POSIX and mostly atomic on Windows
        os.replace(temp_file, self.storage_file)
        
        print("💾 (Background) Graph state saved atomically.")

    def load_graph(self):
        with open(self.storage_file, 'rb') as f:
            self.G = pickle.load(f)
        print("📂 (Startup) Graph state loaded from pickle.")
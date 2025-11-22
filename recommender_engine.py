import networkx as nx
import numpy as np
import pickle
import os
import math
import threading
import time
import random
import torch
import faiss  # <--- The Engine for Scalability
from datetime import datetime
from sentence_transformers import SentenceTransformer
from functools import lru_cache

class RobustRecommender:
    def __init__(self, papers_data, model_name='all-MiniLM-L6-v2', storage_file='graph_state.pkl'):
        """
        Scalable Architecture using FAISS + Time-Based Gravity.
        """
        self.papers = papers_data
        self.paper_map = {str(p['id']): p for p in papers_data}
        self.paper_ids = [str(p['id']) for p in papers_data]
        self.storage_file = storage_file
        
        # --- THREAD LOCK ---
        self._lock = threading.Lock() 
        
        # --- HYPERPARAMETERS ---
        self.ALPHA = 0.6        # Semantic Similarity
        self.BETA = 0.3         # Popularity
        self.GAMMA = 0.1        # Freshness
        
        # --- GRAVITY PARAMETERS ---
        self.GRAVITY = 1.8      
        self.BASE_BOOST = 1.0   
        self.LEARNING_RATE = 0.5 
        
        # Map ID to Index for Vector Lookups
        self.paper_id_to_idx = {pid: i for i, pid in enumerate(self.paper_ids)}
        
        # Load Model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"⏳ Loading Model on {device.upper()}...")
        self.model = SentenceTransformer(model_name, device=device)
        
        # Load Embeddings & Build FAISS Index
        self._init_embeddings()

        # Initialize Graph
        if os.path.exists(self.storage_file):
            self.load_graph()
            self._sync_popularity_from_db()
        else:
            self.build_cold_start_graph()

    def _init_embeddings(self):
        """
        Loads embeddings, Normalizes them, and Builds FAISS Index.
        """
        embeddings_file = 'paper_embeddings.pkl'
        
        # 1. Load or Compute
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                self.item_embeddings = pickle.load(f)
        else:
            print("🔍 Computing embeddings...")
            texts = [f"{p['title']} {p['summary']}" for p in self.papers]
            self.item_embeddings = self.model.encode(texts, batch_size=128, show_progress_bar=True)
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.item_embeddings, f)
        
        # 2. Prepare for FAISS (Critical Steps)
        # FAISS requires float32. 
        self.item_embeddings = self.item_embeddings.astype('float32')
        
        # Normalize L2 (Euclidean) so that Inner Product == Cosine Similarity
        faiss.normalize_L2(self.item_embeddings)
        
        print(f"✅ Embeddings ready: {self.item_embeddings.shape}")

        # 3. Build FAISS Index
        # IndexFlatIP: Exact Search via Inner Product. 
        # Fast enough for <1M items, extremely memory efficient compared to n^2 matrix.
        self.dimension = self.item_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.item_embeddings)
        print(f"🚀 FAISS Index built with {self.index.ntotal} vectors.")

    def build_cold_start_graph(self, k=4):
        """
        SCALABLE VERSION: O(N * k)
        Replaces the O(N^2) matrix multiplication.
        """
        print("🆕 Building Cold Start Graph using FAISS...")
        self.G = nx.Graph()
        current_time = time.time()

        # 1. Batch Search for ALL items at once
        # We ask for k+1 because the closest neighbor is always the item itself
        # D = Distances (Scores), I = Indices
        D, I = self.index.search(self.item_embeddings, k + 1)

        # 2. Construct Graph
        for i, paper_id in enumerate(self.paper_ids):
            paper = self.paper_map[paper_id]
            
            # Add Node
            self.G.add_node(paper_id, 
                          hits=paper.get('popularity', 0),
                          title=paper['title'],
                          published=paper.get('published', ''))
            
            # Add Edges (from FAISS results)
            # D[i] is list of scores, I[i] is list of neighbor indices
            for rank, neighbor_idx in enumerate(I[i]):
                if neighbor_idx == i: continue # Skip self
                if neighbor_idx == -1: continue # Padding check

                neighbor_pid = self.paper_ids[neighbor_idx]
                sim_score = float(D[i][rank])
                
                # Clip score for safety
                sim_score = max(0.1, min(sim_score, 1.0))
                
                self.G.add_edge(paper_id, neighbor_pid, 
                                raw_weight=sim_score * 5.0, 
                                last_updated=current_time)

    def _get_effective_weight(self, raw_weight, last_updated_ts):
        """
        THE GRAVITY FORMULA
        """
        hours_passed = (time.time() - last_updated_ts) / 3600.0
        denominator = pow(hours_passed + 2, self.GRAVITY)
        return raw_weight / denominator

    def _calculate_hybrid_score(self, similarity, node_data):
        # 1. Popularity (Log smoothed)
        hits = node_data.get('hits', 0)
        pop_score = min(math.log1p(hits) / 10.0, 1.0)
        
        # 2. Recency
        recency_score = 0
        if 'published' in node_data and node_data['published']:
            try:
                pub_date = datetime.fromisoformat(node_data['published'].replace('Z', '+00:00'))
                days_old = (datetime.now(pub_date.tzinfo) - pub_date).days
                recency_score = max(0, 1 - (days_old / 365.0))
            except: pass

        return (self.ALPHA * similarity) + (self.BETA * pop_score) + (self.GAMMA * recency_score)

    def search(self, query, current_paper_id=None, num_recs=5):
        """
        Multi-Source Retrieval Search (Scalable Version).
        Uses FAISS for Semantic Source + NetworkX for Behavioral Source.
        """
        # 1. Encode Query & Normalize
        query_emb = self.model.encode([query])[0].astype('float32')
        faiss.normalize_L2(query_emb.reshape(1, -1))
        
        # ---------------------------------------------------------
        # 2. SOURCE A: SEMANTIC CANDIDATES (Via FAISS)
        # ---------------------------------------------------------
        sem_pool_size = 50
        D, I = self.index.search(np.array([query_emb]), sem_pool_size)
        
        sem_indices = I[0]
        sem_scores = D[0]
        
        # Create a fast lookup map: {paper_id: similarity_score}
        # This prevents re-calculating dot products for top hits
        known_sims = {}
        for idx, score in zip(sem_indices, sem_scores):
            if idx != -1:
                pid = self.paper_ids[idx]
                known_sims[pid] = float(score)

        # ---------------------------------------------------------
        # 3. SOURCE B: BEHAVIORAL CANDIDATES (Via Graph)
        # ---------------------------------------------------------
        graph_indices = []
        context_boosts = {}
        
        if current_paper_id:
            str_id = str(current_paper_id)
            if self.G.has_node(str_id):
                neighbors = list(self.G.neighbors(str_id))
                
                # Calculate Effective Weights
                neighbor_weights = []
                for nbr in neighbors:
                    edge = self.G[str_id][nbr]
                    w = self._get_effective_weight(edge['raw_weight'], edge['last_updated'])
                    neighbor_weights.append((nbr, w))
                
                # Top 20 Strongest Links
                neighbor_weights.sort(key=lambda x: x[1], reverse=True)
                top_neighbors = neighbor_weights[:20]
                
                for nbr, w in top_neighbors:
                    context_boosts[nbr] = w
                    # Add to candidate pool
                    if nbr not in known_sims:
                        # Only add to graph_indices if FAISS didn't already find it
                        try:
                            idx = self.paper_id_to_idx.get(nbr)
                            graph_indices.append(idx)
                        except ValueError: pass

        # ---------------------------------------------------------
        # 4. THE MERGE & SCORING
        # ---------------------------------------------------------
        # Combine FAISS results + Graph results
        all_candidates_idx = list(sem_indices) + list(graph_indices)
        all_candidates_idx = list(set(all_candidates_idx)) # Unique
        
        candidates = []
        
        for idx in all_candidates_idx:
            if idx == -1: continue
            pid = self.paper_ids[idx]
            
            if pid not in self.G.nodes: continue
            
            # --- LAZY SIMILARITY CALCULATION ---
            # If pid was in FAISS results, we have the score.
            # If pid came from Graph, we must calculate similarity now (Lazy Load)
            if pid in known_sims:
                sim_score = known_sims[pid]
            else:
                # We have to compute dot product manually for this specific item
                # This is very fast (only happens for ~20 items max)
                item_emb = self.item_embeddings[idx]
                sim_score = float(np.dot(query_emb, item_emb))
                sim_score = max(0.0, sim_score)

            # A. Base Score
            base_score = self._calculate_hybrid_score(sim_score, self.G.nodes[pid])
            
            # B. Apply Boost
            boost = context_boosts.get(pid, 0.0) * 0.5
            
            final_score = base_score + boost
            candidates.append((pid, final_score))

        # ---------------------------------------------------------
        # 5. FINAL SORT & RETURN
        # ---------------------------------------------------------
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for pid, score in candidates[:num_recs]:
            p_data = self.paper_map[pid]
            results.append({
                'id': pid,
                'title': p_data['title'],
                'score': round(score, 4),
                'popularity': self.G.nodes[pid].get('hits', 0),
                'authors': p_data.get('authors', ''),
                'summary': p_data.get('summary', ''),
                "pdf_url": p_data.get("pdf_url")
            })
            
        return results

    def process_user_click(self, query, clicked_item):
        """
        OPTIMIZED WRITE PATH.
        """
        if clicked_item not in self.G.nodes: return

        # 1. Sim Calculation (Outside Lock)
        query_emb = self.model.encode([query])[0].astype('float32')
        faiss.normalize_L2(query_emb.reshape(1, -1))
        
        # Find the "Start Node" (The item most similar to the query)
        # Using FAISS for O(log N) speed instead of argmax over all items
        D, I = self.index.search(np.array([query_emb]), 1)
        start_node_idx = I[0][0]
        start_node = self.paper_ids[start_node_idx]

        current_time = time.time()

        # 2. Atomic Update (Inside Lock)
        with self._lock:
            self.G.nodes[clicked_item]['hits'] += 1
            
            if start_node != clicked_item:
                if self.G.has_edge(start_node, clicked_item):
                    data = self.G[start_node][clicked_item]
                    data['raw_weight'] += self.LEARNING_RATE
                    data['last_updated'] = current_time
                else:
                    self.G.add_edge(start_node, clicked_item, 
                                  raw_weight=self.BASE_BOOST, 
                                  last_updated=current_time)

    def get_similar_items(self, paper_id, num_recs=3):
        """
        Beam Search with Semantic & Weight Priority.
        """
        if paper_id not in self.G.nodes: return []
        
        # 1. Get Start Node Embedding
        try:
            start_idx = self.paper_id_to_idx[paper_id]
            start_emb = self.item_embeddings[start_idx]
        except (ValueError, KeyError):
            start_emb = None

        frontier_pool = {}
        visited = {paper_id}
        results = []
        
        # Step 1: Initialize
        self._expand_frontier(paper_id, frontier_pool, visited, start_emb, hop_penalty=1.0)
        
        # Step 2: Loop
        for _ in range(num_recs):
            if not frontier_pool: break
            
            candidates = [
                (pid, s) for pid, s in frontier_pool.items() 
                if pid not in visited
            ]
            
            if not candidates: break
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Selection Logic
            candidate_ids = [c[0] for c in candidates]
            scores = np.array([c[1] for c in candidates])
            
            if scores.sum() == 0:
                chosen_pid = random.choice(candidate_ids)
            else:
                if random.random() < 0.6:
                    chosen_pid = candidate_ids[0]
                else:
                    top_k = min(len(scores), 5)
                    probs = scores[:top_k] / scores[:top_k].sum()
                    chosen_pid = np.random.choice(candidate_ids[:top_k], p=probs)
            
            visited.add(chosen_pid)
            if chosen_pid in self.paper_map:
                p_data = self.paper_map[chosen_pid].copy()
                p_data['score'] = round(frontier_pool[chosen_pid], 4)
                p_data['id'] = str(p_data['id'])
                results.append(p_data)
            
            # Expand
            self._expand_frontier(chosen_pid, frontier_pool, visited, start_emb, hop_penalty=0.6)

        return results

    def _expand_frontier(self, current_node, pool, visited, start_emb, hop_penalty):
        for n in self.G.neighbors(current_node):
            if n in visited: continue
            
            graph_score = self._calculate_graph_score(current_node, n)
            
            semantic_score = 0.5 
            if start_emb is not None:
                try:
                    n_idx = self.paper_id_to_idx[n]
                    n_emb = self.item_embeddings[n_idx]
                    # Quick dot product
                    semantic_score = float(np.dot(start_emb, n_emb))
                    semantic_score = max(0.0, semantic_score)
                except (ValueError, KeyError): pass

            final_score = (graph_score * (1 + semantic_score)) * hop_penalty
            
            if n not in pool or final_score > pool[n]:
                pool[n] = final_score

    def _calculate_graph_score(self, u, v):
        if not self.G.has_edge(u, v): return 0.0
        
        edge = self.G[u][v]
        node_v = self.G.nodes[v]
        
        r_weight = edge.get('raw_weight', edge.get('weight', 1.0) * 5.0)
        l_updated = edge.get('last_updated', time.time())
        
        eff_weight = self._get_effective_weight(r_weight, l_updated)
        
        dest_hits = node_v.get('hits', 0)
        pop_multiplier = math.log1p(dest_hits) + 1.0
        
        return eff_weight * pop_multiplier

    def get_trending(self, limit=5):
        """
        Returns top papers purely by Global Popularity (Hits).
        
        OPTIMIZATION: 
        Uses heapq.nlargest instead of sorted().
        - sorted() is O(N log N) -> Slow for large graphs.
        - heapq.nlargest() is O(N log K) -> Much faster when limit is small (e.g. 5).
        """
        import heapq

        # Efficiently grab top N items without sorting the whole list
        # self.G.nodes(data=True) returns an iterator (id, data_dict)
        top_nodes = heapq.nlargest(
            limit, 
            self.G.nodes(data=True), 
            key=lambda x: x[1].get('hits', 0)
        )
        
        results = []
        for paper_id, node_data in top_nodes:
            # Safety check in case graph has nodes not in DB map
            if paper_id not in self.paper_map: continue
                
            p_data = self.paper_map[paper_id]
            
            results.append({
                'id': str(paper_id),
                'title': p_data['title'],
                'score': 0, # Score is the raw hit count
                'popularity': node_data.get('hits', 0),
                'authors': p_data.get('authors', ''),
                'summary': p_data.get('summary', ''),
                'published': p_data.get('published', ''),
                "pdf_url": p_data.get("pdf_url")
            })
            
        return results

    def save_to_disk(self):
        temp = self.storage_file + ".tmp"
        with self._lock:
            with open(temp, 'wb') as f: pickle.dump(self.G, f)
        os.replace(temp, self.storage_file)

    def load_graph(self):
        with open(self.storage_file, 'rb') as f: self.G = pickle.load(f)
    
    def _sync_popularity_from_db(self):
        with self._lock:
            for pid, data in self.paper_map.items():
                if pid in self.G.nodes: self.G.nodes[pid]['hits'] = data['popularity']
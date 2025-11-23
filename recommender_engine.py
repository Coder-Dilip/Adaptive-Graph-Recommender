import numpy as np
import math
import time
import random
import uuid
import torch
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, FieldCondition
from sentence_transformers import SentenceTransformer
HAS_LOCAL_MODEL = True

class RobustRecommender:
    def __init__(self, papers_data, 
                 qdrant_url=None, 
                 qdrant_api_key=None
                 ):
        """
        Remote Architecture:
        1. Qdrant Cloud (Offloads Storage/Search RAM)
        """
        print("\n[RobustRecommender] Initializing...")
        self.papers = papers_data
        self.paper_map = {str(p['id']): p for p in papers_data}
        self.collection_name = "papers_production"
        
        # --- PRESERVED LOGIC PARAMS ---
        self.ALPHA = 0.6
        self.BETA = 0.3
        self.GAMMA = 0.1
        self.GRAVITY = 1.8      
        self.BASE_BOOST = 1.0   
        self.LEARNING_RATE = 0.5 

        # --- 1. CONNECT TO REMOTE QDRANT ---
        print("\n[RobustRecommender] Setting up Qdrant client...")
        try:
            if qdrant_url and qdrant_api_key:
                print(f"   - Connecting to Qdrant Cloud at {qdrant_url}...")
                self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=10.0)
                print("   - Successfully connected to Qdrant Cloud")
            else:
                print("   - Connecting to local Qdrant instance...")
                self.client = QdrantClient(host="localhost", port=6333, timeout=10.0)
                print("   - Successfully connected to local Qdrant")
        except Exception as e:
            print(f"   - Failed to connect to Qdrant: {str(e)}")
            raise

        # --- 2. SETUP EMBEDDING MODEL ---
        print("\n[RobustRecommender] Setting up embedding model...")
        self.openai_client = None
        self.local_model = None
        
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"   - Using device: {device}")
            print("   - Loading 'all-MiniLM-L6-v2' model (this may take a minute)...")
            
            # Enable progress bar for model loading
            import logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2', 
                                                device=device,
                                                cache_folder='./models')
            
            self.dimension = 384
            self.use_cloud_embeddings = False
            print("   - Model loaded successfully")
            
        except Exception as e:
            print(f"   - Failed to load model: {str(e)}")
            raise

        # Ensure DB Exists
        self._init_qdrant_collection()
        
        # Check Cold Start
        if self.client.count(self.collection_name).count < len(self.papers):
            self.build_cold_start_graph()
        else:
            print("✅ System Ready (Remote Data Loaded).")

    def _get_embedding(self, text):
        """Wrapper to switch between Local and Cloud embeddings"""
        return self.local_model.encode(text).tolist()

    def _get_batch_embeddings(self, texts):
        """Batch wrapper"""
        return self.local_model.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    @property
    def model(self):
        """For backward compatibility with existing code"""
        return self.local_model

    def _init_qdrant_collection(self):
        # 1. Create Collection if it doesn't exist
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
            )
            
        # 2. --- THE FIX: CREATE PAYLOAD INDEX ---
        # We tell Qdrant: "Please optimize searches for 'original_id'"
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="original_id",
            field_schema="keyword"  # <--- Keyword means "Exact Match" (perfect for IDs)
        )
        
        # Optional: Index 'hits' too, so sorting by popularity is fast
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="hits",
            field_schema="integer"
        )
        
        print("✅ Qdrant Collection & Indexes Verified.")

    def _parse_published_date(self, date_str):
        """Helper for Gamma (Freshness) calculation prep."""
        if not date_str: return 0
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return int(dt.timestamp())
        except: return 0

    def _get_effective_weight(self, raw_weight, last_updated_ts):
        """
        THE GRAVITY FORMULA (PRESERVED).
        Decays edge weight based on time since last interaction.
        """
        if last_updated_ts == 0: return raw_weight
        
        hours_passed = (time.time() - last_updated_ts) / 3600.0
        # Formula: weight / (hours + 2)^1.8
        denominator = pow(hours_passed + 2, self.GRAVITY)
        return raw_weight / denominator

    def _calculate_hybrid_score(self, similarity, payload):
        """
        THE HYBRID FORMULA (PRESERVED).
        Combines Semantic Score + Popularity + Freshness.
        """
        # 1. Popularity (Log smoothed) - Hits stored in payload
        hits = payload.get('hits', 0)
        pop_score = min(math.log1p(hits) / 10.0, 1.0)
        
        # 2. Recency
        recency_score = 0
        pub_str = payload.get('raw_published')
        if pub_str:
            try:
                pub_date = datetime.fromisoformat(pub_str.replace('Z', '+00:00'))
                days_old = (datetime.now(pub_date.tzinfo) - pub_date).days
                recency_score = max(0, 1 - (days_old / 365.0))
            except: pass

        return (self.ALPHA * similarity) + (self.BETA * pop_score) + (self.GAMMA * recency_score)

    def build_cold_start_graph(self):
        """
        Replaces NetworkX Build. Indexes Vectors + Metadata + Adjacency Structs.
        """
        print("🔥 Running Cold Start Indexing...")
        texts = [f"{p['title']} {p.get('summary', '')}" for p in self.papers]
        embeddings = self.model.encode(texts, batch_size=128, show_progress_bar=True)
        
        points = []
        for i, paper in enumerate(self.papers):
            pid = str(paper['id'])
            # Qdrant needs Int or UUID. We use UUID generated from string ID.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, pid))

            payload = {
                "original_id": pid,
                "title": paper['title'],
                "hits": paper.get('popularity', 0),
                "raw_published": paper.get('published', ''),
                "authors": paper.get('authors', ''),
                "summary": paper.get('summary', ''),
                "pdf_url": paper.get('pdf_url'),
                
                # --- THE WORMHOLE DATA STRUCTURE ---
                # Stores links to other papers. 
                # Format: { "target_id": { "w": float, "ts": timestamp } }
                "behavioral_links": {} 
            }

            points.append(PointStruct(id=point_id, vector=embeddings[i].tolist(), payload=payload))

        # Batch Upload
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(self.collection_name, points[i : i + batch_size])
        print("✅ Indexing Complete.")

    def search(self, query, current_paper_id=None, num_recs=5):
        """
        Merges Vector Search (Source A) + Graph Neighbors (Source B).
        """
        # 1. Encode Query
        query_vector = self.model.encode(query).tolist()

        # 2. Source A: Semantic Search (Qdrant)
        sem_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=50,
            with_payload=True
        ).points

        # Map {id: point} for easy merging
        candidates = {hit.payload['original_id']: {'point': hit, 'boost': 0.0} for hit in sem_hits}

        # 3. Source B: Behavioral Context (The "Rescue")
        if current_paper_id:
            curr_point = self._get_point_by_original_id(current_paper_id)
            if curr_point and 'behavioral_links' in curr_point.payload:
                links = curr_point.payload['behavioral_links']
                
                # Get IDs of strong links
                strong_link_ids = [pid for pid in links.keys()]
                
                if strong_link_ids:
                    # Fetch these specific papers (Batch Fetch)
                    graph_hits = self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=Filter(
                            must=[FieldCondition(key="original_id", match={"any": strong_link_ids})]
                        ),
                        limit=20,
                        with_payload=True,
                        with_vectors=True # Need vector to calc sim manually
                    )[0]

                    for hit in graph_hits:
                        pid = hit.payload['original_id']
                        
                        # Calculate Gravity Weight
                        l_data = links[pid]
                        eff_weight = self._get_effective_weight(l_data['w'], l_data['ts'])
                        
                        # Add to candidates (or update boost if already there)
                        if pid not in candidates:
                            # We need to manually calc semantic score for sorting later
                            # Dot product of query_vector AND hit.vector
                            # (Assuming normalized vectors, dot = cosine)
                            sim = np.dot(query_vector, hit.vector)
                            hit.score = sim # Manually assign score
                            candidates[pid] = {'point': hit, 'boost': 0.0}
                        
                        # APPLY THE BOOST
                        candidates[pid]['boost'] = eff_weight * 0.5

        # 4. Scoring & Sorting
        final_list = []
        for pid, data in candidates.items():
            point = data['point']
            boost = data['boost']
            
            # Re-calculate hybrid score using point.score (Semantic) + point.payload (Popularity)
            # Note: For Graph-only hits, point.score was calculated manually above
            hits = point.payload.get('hits', 0)
            pop_score = min(math.log1p(hits) / 10.0, 1.0)
            
            base_score = (0.6 * point.score) + (0.3 * pop_score) # Simplified Hybrid
            total_score = base_score + boost
            
            final_list.append({
                'id': pid,
                'title': point.payload['title'],
                'score': round(total_score, 4),
                'popularity': point.payload['hits'],
                "authors": point.payload['authors'],
                "summary": point.payload['summary']
            })

        final_list.sort(key=lambda x: x['score'], reverse=True)
        return final_list[:num_recs]

    def process_user_click(self, query, clicked_item_id):
        """
        Updates the 'behavioral_links' payload inside Qdrant.
        """
        current_time = time.time()
        
        # 1. Find Context (Start Node)
        query_vector = self.model.encode(query).tolist()
        search_res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=1,
            with_payload=True
        ).points

        if not search_res: return
        start_node = search_res[0]
        start_pid = start_node.payload['original_id']

        if str(start_pid) == str(clicked_item_id): return

        # 2. Update The Payload Graph
        current_links = start_node.payload.get("behavioral_links", {})
        clicked_str = str(clicked_item_id)
        
        # Logic: Get Old -> Add Learning Rate -> Set New
        link_data = current_links.get(clicked_str, {'w': 0.0, 'ts': 0})
        # Handle case where Qdrant returns non-dict
        if not isinstance(link_data, dict): link_data = {'w': 0.0, 'ts': 0}

        new_weight = link_data['w'] + self.LEARNING_RATE if link_data['w'] > 0 else self.BASE_BOOST
        
        current_links[clicked_str] = {
            'w': new_weight,
            'ts': current_time
        }

        # Optimization: Limit payload size (Keep Top 50 links)
        if len(current_links) > 50:
            current_links = dict(sorted(current_links.items(), key=lambda x: x[1]['w'], reverse=True)[:50])

        # 3. Push Update to Qdrant
        # Note: In Qdrant, we must overwrite the full payload field
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"behavioral_links": current_links},
            points=[start_node.id]
        )
        
        # 4. Increment Popularity on Target
        target_point = self._get_point_by_original_id(clicked_item_id)
        if target_point:
            new_hits = target_point.payload.get('hits', 0) + 1
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"behavioral_links": current_links},
                points=[start_node.id]
            )
            
            # 4. Increment Popularity on Target
            target_point = self._get_point_by_original_id(clicked_item_id)
            if target_point:
                new_hits = target_point.payload.get('hits', 0) + 1
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={"hits": new_hits},
                    points=[target_point.id]
                )

        # --- HELPERS ---
    def _get_point_by_original_id(self, original_id):
            """Helper to get a point by its original_id (not Qdrant's internal ID)
            
            Handles both string and integer IDs by trying both if needed.
            """
            if original_id is None:
                return None
                
            # Try with the ID as provided first
            res = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=[FieldCondition(key="original_id", match={"value": original_id})]),
                limit=1,
                with_payload=True,
                with_vectors=True
            )[0]
            
            # If not found, try converting the ID type
            if not res and isinstance(original_id, str) and original_id.isdigit():
                try:
                    res = self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=Filter(must=[FieldCondition(key="original_id", match={"value": int(original_id)})]),
                        limit=1,
                        with_payload=True,
                        with_vectors=True
                    )[0]
                except (ValueError, TypeError):
                    pass
                    
            return res[0] if res else None

    def get_similar_items(self, paper_id, num_recs=3):
        """
        Robust Beam Search with Cold-Start Fallback.
        """
        # 1. Fetch Start Node
        start_point = self._get_point_by_original_id(paper_id)
        if not start_point: 
            print(f"⚠️ Paper ID {paper_id} not found in Qdrant.")
            return []
        
        start_vector = np.array(start_point.vector) if start_point.vector else None

        # The Frontier: {paper_id: score}
        frontier_pool = {}
        visited = {str(paper_id)}
        results = []

        # --- PHASE 1: TRY GRAPH WALK (Business Intent) ---
        # Fetch Level 1 Neighbors from the Payload Graph
        self._expand_frontier_from_payload(start_point, frontier_pool, visited, start_vector, hop_penalty=1.0)

        # --- PHASE 2: FALLBACK (If Graph is empty) ---
        # If nobody has clicked connections from this paper yet, frontier_pool will be empty.
        # We MUST switch to Vector Search so we don't return 404.
        if not frontier_pool:
            print(f"ℹ️ No behavioral links for {paper_id}. Falling back to Semantic Search.")
            return self._get_semantic_recommendations(start_point.id, num_recs)

        # --- PHASE 3: THE BEAM LOOP (If Graph exists) ---
        for _ in range(num_recs):
            if not frontier_pool: break

            # A. Filter & Sort
            candidates = [(pid, s) for pid, s in frontier_pool.items() if pid not in visited]
            if not candidates: break
            candidates.sort(key=lambda x: x[1], reverse=True)

            # B. 70/30 Logic
            candidate_ids = [c[0] for c in candidates]
            scores = np.array([c[1] for c in candidates])
            
            chosen_pid = None
            if scores.sum() == 0:
                chosen_pid = random.choice(candidate_ids)
            else:
                if random.random() < 0.7:
                    chosen_pid = candidate_ids[0] # Deterministic
                else:
                    top_k = min(len(scores), 5)
                    probs = scores[:top_k] / scores[:top_k].sum()
                    chosen_pid = np.random.choice(candidate_ids[:top_k], p=probs) # Exploratory

            # C. Fetch Winner Data (Network Call)
            winner_point = self._get_point_by_original_id(chosen_pid)
            if not winner_point: continue 

            # D. Add to Results
            visited.add(str(chosen_pid))
            p_data = {
                'id': winner_point.payload['original_id'],
                'title': winner_point.payload['title'],
                'score': round(frontier_pool[chosen_pid], 4),
                'popularity': winner_point.payload['hits'],
                'authors': winner_point.payload.get('authors', ''),
                'summary': winner_point.payload.get('summary', ''),
                'pdf_url': winner_point.payload.get('pdf_url', '')
            }
            results.append(p_data)

            # E. EXPAND (Level 2)
            self._expand_frontier_from_payload(winner_point, frontier_pool, visited, start_vector, hop_penalty=0.6)

        # If Beam search ran out of ideas but we still need more recs, fill with semantic
        if len(results) < num_recs:
            needed = num_recs - len(results)
            semantic_fillers = self._get_semantic_recommendations(start_point.id, needed, exclude_ids=visited)
            results.extend(semantic_fillers)

        return results

    def _get_semantic_recommendations(self, positive_point_id, limit, exclude_ids=None):
        """Fallback: Standard Qdrant Vector Recommendation"""
        try:
            # Create filter to exclude items we already have
            query_filter = None
            if exclude_ids:
                # Qdrant expects internal IDs or payload matches. 
                # Simplest is to just fetch more and filter in Python, but let's try payload filter.
                should_exclude = [FieldCondition(key="original_id", match={"value": str(eid)}) for eid in exclude_ids]
                query_filter = Filter(must_not=should_exclude)

            recs = self.client.recommend(
                collection_name=self.collection_name,
                positive=[positive_point_id],
                limit=limit,
                query_filter=query_filter,
                with_payload=True
            )
            
            results = []
            for hit in recs:
                results.append({
                    'id': hit.payload['original_id'],
                    'title': hit.payload['title'],
                    'score': round(hit.score, 4), # Vector score
                    'popularity': hit.payload.get('hits', 0),
                    'authors': hit.payload.get('authors', ''),
                    'summary': hit.payload.get('summary', ''),
                    'pdf_url': hit.payload.get('pdf_url', '')
                })
            return results
        except Exception as e:
            print(f"Semantic Fallback failed: {e}")
            return []

    def _expand_frontier_from_payload(self, point, pool, visited, start_vector, hop_penalty):
            """
            Reads 'behavioral_links' from Qdrant Payload and scores them.
            """
            links = point.payload.get('behavioral_links', {})
            
            # Pre-fetch vectors for these links if we want strict semantic checking?
            # For performance in Remote DB, we might skip strict vector dot-product for every neighbor 
            # unless we fetch them in batch. 
            # HERE, we trust the Graph Weight primarily, but apply a base semantic checks if available.

            for target_id, link_data in links.items():
                if target_id in visited: continue

                # 1. Graph Score (Gravity Applied)
                raw_w = link_data.get('w', 0)
                ts = link_data.get('ts', 0)
                graph_score = self._get_effective_weight(raw_w, ts)

                # 2. Semantic Score (Approximation or Skip)
                # In a remote architecture, fetching vectors for every neighbor is slow (N+1 problem).
                # We assume if the link exists, it has *some* relevance.
                # However, if 'Programming -> Music' has huge weight, graph_score will be huge (e.g., 50.0).
                # Even if semantic_multiplier is 1.0, 50.0 wins.
                
                semantic_multiplier = 1.0 # Default neutral
                
                # Final Score
                final_score = (graph_score * semantic_multiplier) * hop_penalty

                if target_id not in pool or final_score > pool[target_id]:
                    pool[target_id] = final_score

    def get_trending(self, limit=5):
            """
            Returns top papers by Global Popularity.
            """
            # Qdrant Scroll (Scan) - Efficient for getting top items if we sort in app
            # For huge datasets, you would add a payload index on 'hits'
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )[0]
            
            results.sort(key=lambda x: x.payload.get('hits', 0), reverse=True)
            
            output = []
            for hit in results[:limit]:
                output.append({
                    'id': hit.payload['original_id'],
                    'title': hit.payload['title'],
                    'popularity': hit.payload['hits'],
                    'authors': hit.payload['authors'],
                    'pdf_url': hit.payload['pdf_url'],
                    'summary': hit.payload['summary'],
                    'published': hit.payload['published'],
                    'score': 0  # Add missing score field
                })
            return output
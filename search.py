"""Search functionality for segments - keyword and semantic."""

import numpy as np
from embedding_model import encode_sentences

class SegmentSearch:
    """Search segments using keyword matching and semantic similarity."""
    
    def __init__(self, segment_index):
        """Initialize with a SegmentIndex instance."""
        self.index = segment_index
    
    def keyword_search(self, query, top_k=5):
        """Search segments by keyword matching.
        
        Args:
            query: Search query string
            top_k: Number of results to return
        
        Returns:
            List of matching segments sorted by relevance
        """
        if not query or len(query.strip()) == 0:
            return []
        
        query_lower = query.lower()
        results = []
        
        for seg in self.index.segments:
            score = 0
            
            # Check keywords
            if any(query_lower in kw.lower() for kw in seg["keywords"]):
                score += 3
            
            # Check text
            text_content = seg.get("segments", seg.get("text", ""))
            if query_lower in text_content.lower():
                word_count = len(text_content.split())
                occurrences = text_content.lower().count(query_lower)
                score += max(1, occurrences / max(1, word_count / 10))
            
            if score > 0:
                results.append((seg, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return [seg for seg, _ in results[:top_k]]
    
    def semantic_search(self, query, top_k=5, threshold=0.3):
        """Search segments using semantic similarity.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            threshold: Minimum similarity score to include
        
        Returns:
            List of semantically similar segments with scores
        """
        if not query or len(query.strip()) == 0:
            return []
        
        try:
            # Encode query
            query_embedding = encode_sentences([query])[0]
            
            # Ensure it's a numpy array
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
            
            results = []
            for seg in self.index.segments:
                if seg["embedding"] is None:
                    continue
                
                # Convert embedding to numpy array if needed
                if isinstance(seg["embedding"], list):
                    seg_embedding = np.array(seg["embedding"])
                else:
                    seg_embedding = seg["embedding"]
                
                # Compute cosine similarity
                similarity = np.dot(query_embedding, seg_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(seg_embedding) + 1e-8
                )
                
                # Convert to Python float
                similarity = float(similarity)
                
                if similarity >= threshold:
                    results.append((seg, similarity))
            
            # Sort by similarity descending
            results.sort(key=lambda x: x[1], reverse=True)
            return [(seg, score) for seg, score in results[:top_k]]
        
        except Exception as e:
            print(f"Warning: Semantic search failed: {str(e)}")
            return []
    
    def combined_search(self, query, top_k=5, keyword_weight=0.4, semantic_weight=0.6):
        """Combine keyword and semantic search for best results.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            keyword_weight: Weight for keyword search (0-1)
            semantic_weight: Weight for semantic search (0-1)
        
        Returns:
            List of best matching segments with combined scores
        """
        if not query or len(query.strip()) == 0:
            return []
        
        try:
            # Get keyword search results
            keyword_results = self.keyword_search(query, top_k=len(self.index.segments))
            keyword_scores = {seg["id"]: 1.0 - (i / max(1, len(keyword_results))) 
                             for i, seg in enumerate(keyword_results)}
            
            # Get semantic search results
            semantic_results = self.semantic_search(query, top_k=len(self.index.segments), threshold=0.1)
            semantic_scores = {seg["id"]: float(score) 
                              for seg, score in semantic_results}
            
            # Combine scores
            combined_scores = {}
            for seg in self.index.segments:
                kw_score = keyword_scores.get(seg["id"], 0.0)
                sem_score = semantic_scores.get(seg["id"], 0.0)
                combined = float((kw_score * keyword_weight) + (sem_score * semantic_weight))
                combined_scores[seg["id"]] = combined
            
            # Sort and return top_k
            sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            results = []
            for seg_id, score in sorted_ids[:top_k]:
                if score > 0:
                    seg = self.index.get_segment_by_id(seg_id)
                    if seg:
                        results.append((seg, score))
            
            return results
        
        except Exception as e:
            print(f"Warning: Combined search failed: {str(e)}")
            return []

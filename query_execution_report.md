# CogInfera Query Execution Report

**Query Evaluated:** `"whats a deadlock and how to prevent it"`
**Source Material:** *Operating Systems: Three Easy Pieces* (609 Pages, 560 indexed chunks)
**Base LLM:** `qwen/qwen3.5-9b` (via OpenRouter, reasoning enabled)

This document is a complete trace of how the Agentic Hierarchical Graph-Augmented Multi-Hop RAG System processed the user's query from start to finish.

---

## 1. Query Planning & Decomposition ([QueryPlanner](file:///c:/Users/Rituraj%20Singh/Documents/arc%20proj_py/CogInfera/agents/query_planner.py#27-55))
**Goal:** Analyze the complexity of the raw user query and break it down into atomic, independent sub-queries if it requires multi-hop reasoning.

- **Input passed to LLM:** `"whats a deadlock and how to prevent it"`
- **LLM Reasoning:** The model recognized that understanding a deadlock and understanding its prevention strategies are two distinct operating system concepts that might be located in different sections of the textbook.
- **LLM Output (JSON):** 
  ```json
  {
    "complexity": "multi_hop",
    "sub_queries": [
      "What is a deadlock in computer science?",
      "What are common prevention strategies for deadlocks in software systems?"
    ]
  }
  ```

---

## 2. Agentic Retrieval ([RetrievalController](file:///c:/Users/Rituraj%20Singh/Documents/arc%20proj_py/CogInfera/agents/retrieval_controller.py#26-113) & [HybridRetriever](file:///c:/Users/Rituraj%20Singh/Documents/arc%20proj_py/CogInfera/retrieval/hybrid.py#12-75))
**Goal:** Gather the most relevant chunks of text from the source material for each sub-query, using a mix of vector semantics, exact keyword matching, and cross-encoder reranking.

- **Input:** Two sub-queries from the Query Planner.
- **Process for Sub-Query 1 (What is a deadlock?):** 
  - [DenseSearch](file:///c:/Users/Rituraj%20Singh/Documents/arc%20proj_py/CogInfera/retrieval/dense_search.py#12-66) (FAISS) embedded the query via `BAAI/bge-small-en` and pulled the top 20 semantic matches.
  - [SparseSearch](file:///c:/Users/Rituraj%20Singh/Documents/arc%20proj_py/CogInfera/retrieval/sparse_search.py#12-61) (BM25) tokenized the query and pulled the top 20 exact keyword matches.
  - [HybridRetriever](file:///c:/Users/Rituraj%20Singh/Documents/arc%20proj_py/CogInfera/retrieval/hybrid.py#12-75) merged the two lists using Reciprocal Rank Fusion (RRF).
  - [Reranker](file:///c:/Users/Rituraj%20Singh/Documents/arc%20proj_py/CogInfera/retrieval/reranker.py#9-36) (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scored the merged list against the sub-query and returned the top 10 chunks.
- **Process for Sub-Query 2 (Prevention strategies?):** 
  - Same hybrid retrieval process executed for the prevention strategies.
- **Final Output:** The [RetrievalController](file:///c:/Users/Rituraj%20Singh/Documents/arc%20proj_py/CogInfera/agents/retrieval_controller.py#26-113) deduplicated and combined the results, yielding **17 highly relevant chunks** of text (mostly from Chapters covering Concurrency and Deadlocks, specifically pages 396–407). 

---

## 3. Context Compression ([ContextCompressor](file:///c:/Users/Rituraj%20Singh/Documents/arc%20proj_py/CogInfera/reasoning/context_compressor.py#22-58))
**Goal:** Take the 17 raw chunks (which may contain irrelevant surrounding text) and ask the LLM to discard the noise, keeping only the facts relevant to deadlocks and their prevention.

- **Input passed to LLM:** The user's original query + the 17 raw chunks formatted with their page numbers and section headers.
- **LLM Output:** A single, dense block of text summarizing the core mechanics of deadlocks (Thread 1 waiting for L2 while holding L1, etc.), prevention via lock ordering, avoidance via architecture (MapReduce), and detection mechanisms. All page number citations were preserved in this dense block.

---

## 4. Dynamic Graph Reasoning ([GraphReasoner](file:///c:/Users/Rituraj%20Singh/Documents/arc%20proj_py/CogInfera/reasoning/graph_reasoner.py#32-138))
**Goal:** Extract structured entities and map their relationships mathematically to understand the multi-hop connections.

- **Input passed to LLM:** The dense compressed context block.
- **LLM Output (JSON):** 
  ```json
  {
    "entities": [
      {"name": "Deadlock", "type": "concept", "description": "Threads blocking each other indefinitely"},
      {"name": "Lock Acquisition Total Order", "type": "concept", "description": "Prevention strategy"},
      {"name": "MapReduce", "type": "concept", "description": "Wait-free architecture model"}
    ],
    "relationships": [
      {"source": "Lock Acquisition Total Order", "target": "Deadlock", "relation": "prevents"}
    ]
  }
  ```
- **Process:** The system built an in-memory `NetworkX` directed graph from these nodes and extracted the shortest paths (e.g., `Lock Acquisition Total Order → [prevents] → Deadlock`).
- **Final Output:** A text summary of the graph's multi-hop paths.

---

## 5. Multi-Pass LLM Reasoning ([MultiPassReasoner](file:///c:/Users/Rituraj%20Singh/Documents/arc%20proj_py/CogInfera/reasoning/multi_pass.py#13-105))
**Goal:** Synthesize the final answer using 3 distinct psychological passes to ensure high accuracy and remove hallucination.

### Pass 1: Fact Extraction
- **Input:** The compressed context.
- **LLM Output:** A bulleted list of raw facts (e.g., "Deadlocks account for 31 of 105 concurrency bugs [Page 396]").

### Pass 2: Relationship Mapping
- **Input:** The raw facts + the Graph Reasoner's multi-hop path summary.
- **LLM Output:** A logical map showing that complex locking protocols cause deadlocks, while Total Order and wait-free architectures prevent them.

### Pass 3: Answer Synthesis
- **Input:** The user's query + extracted facts + relationship map.
- **LLM Output:** The beautifully formatted final answer clearly split into "Definition and Mechanics", "Prevention Strategies", "Mitigation", and "Contextual Impact", complete with inline `[[Pages X-Y]]` citations.

---

## 6. Self-RAG Validation ([SelfRAGValidator](file:///c:/Users/Rituraj%20Singh/Documents/arc%20proj_py/CogInfera/reasoning/self_rag.py#32-90))
**Goal:** Have the LLM coldly evaluate its own synthesized answer to ensure it fully answers the prompt and doesn't hallucinate.

- **Input passed to LLM:** Original query, compressed context, and the synthesized final answer.
- **LLM Output (JSON):**
  ```json
  {
    "is_complete": true,
    "is_grounded": true,
    "confidence": 0.95,
    "issues": [],
    "missing_info": "",
    "suggestion": ""
  }
  ```
- **Result:** Because `is_complete`, `is_grounded`, and `confidence` all passed the strict thresholds, the system immediately rendered the final answer to the CLI output instead of triggering a fallback retry.

---

## The Final CLI Output (What the User Saw)

```text
  **Definition and Mechanics** A deadlock occurs in concurrent      
  systems with complex locking protocols when threads block each    
  other indefinitely [[Pages 399-399]]. Researchers identify a cycle
  in the resource graph as an indicator of a deadlock [[Pages       
  399-399]]. A specific mechanical scenario involves Thread 1       
  holding Lock L1 and waiting for L2, while Thread 2 holds L2 and   
  waits for L1 [[Pages 399-399]].  

  **Prevention Strategies** To     
  prevent deadlocks, the following approaches are outlined in the   
  document: 
  *   **Lock Acquisition Total Order:** The best practical
  solution is to be careful and develop a lock acquisition total    
  order, ensuring threads acquire locks in the same sequence [[Pages
  406-407], Section 32.4 Summary]. 
  *   **Avoidance via Architecture:** This can be achieved by avoiding locks entirely   
  through wait-free approaches, though they suffer from complexity  
  and lack of generality [[Pages 406-407]]. Alternatively, models   
  like MapReduce describe parallel computations without any locks,  
  avoiding the deadlock problem by nature [[Pages 406-407]].        

  **Mitigation and Detection** While prevention is preferred, if it 
  is considered too costly, strategies shift to detection and       
  recovery, often resulting in a system restart [[Pages 406-406]].  
  Databases employ periodic deadlock detectors that build resource  
  graphs and check for cycles to identify the issue [[Pages
  406-406]].  

  **Contextual Impact** Deadlocks are a significant     
  category of errors; in a study of modern applications, 31 out of  
  105 total concurrency bugs were classified as deadlocks [[Pages   
  396-396]].
```

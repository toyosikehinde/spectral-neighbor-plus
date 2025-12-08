# **Spectral Neighbor Plus — Contextual Hybrid Music Recommender**

This repository implements a **multimodal music similarity prototype** that integrates lyrical semantics, audio features, and hybrid similarity scoring to model how listeners perceive meaning and sound jointly. The system is built on a curated subset of the 900k-track Spotify dataset containing lyrics, genre labels, emotion metadata, and standardized audio descriptors.

The objective of this prototype is *not* to build a production recommender, but to design and document an **interpretable, experimentally grounded framework** for analyzing multimodal similarity in music recommendation systems.

---

## **Project Overview**

The system constructs three parallel similarity spaces:

---

### **1. Lyrics Semantic Space**

- Lyrics are embedded using a **sentence-transformer model**.  
- Embeddings are normalized into a high-dimensional semantic space that encodes **thematic and emotional meaning**.  
- A **FAISS index** enables efficient nearest-neighbor retrieval.

---

### **2. Audio Feature Space**

Spotify-derived audio descriptors are standardized, normalized, and indexed to model **sonic similarity**:

- energy  
- valence  
- danceability  
- acousticness  
- instrumentalness  
- speechiness  
- liveness  
- tempo  
- loudness  

---

### **3. Hybrid Space**

A linear combination of audio and lyrics cosine similarities forms a **controllable multimodal similarity score**:

\[
s_{\text{hybrid}} = \alpha \cdot s_{\text{audio}} + (1 - \alpha) \cdot s_{\text{lyrics}}
\]

This allows retrieval behavior to shift between:

- **Narrative similarity** (lyrics-driven)  
- **Sonic similarity** (audio-driven)  
- **Balanced multimodal similarity** (hybrid)

---

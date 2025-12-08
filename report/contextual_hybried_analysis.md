# Contextual Hybrid Results Analysis of Lyrics Audio and Multimodal Similarity

## 1 Introduction

This document presents a detailed analysis of the contextual hybrid similarity model developed in the Spectral Neighbor Plus prototype. The objective of this phase of the project was to extend a traditional audio only similarity framework by incorporating contextual semantic information extracted from lyrics thereby modeling music similarity in a way that reflects both how songs sound and what they express narratively. The hybrid system integrates three parallel spaces lyrics semantic space audio feature space and a weighted multimodal hybrid space and evaluates their behaviors using quantitative metrics and qualitative inspection.

## 2 Construction of the Lyrics Semantic Space

Lyrics were embedded using a sentence transformer model and normalized to produce high dimensional semantic representations. The embeddings capture thematic emotional and conceptual content expressed in the text. Each vector encodes meaning beyond simple keyword overlap allowing the model to retrieve songs related by narrative perspective sentiment or subject matter. A FAISS index based on inner product similarity was constructed to enable efficient nearest neighbor retrieval.

Initial qualitative inspection confirmed that nearest neighbors in the semantic space tended to cluster around shared emotional tone or thematic concepts such as heartbreak perseverance introspection or celebration. These results provided evidence that contextual embeddings successfully capture meaning at a level useful for recommendation.

## 3 Construction of the Audio Feature Space

Audio similarity was modeled using standardized Spotify provided audio descriptors including energy danceability valence acousticness instrumentalness speechiness liveness tempo and loudness. Each feature was scaled using a StandardScaler and the resulting vectors were normalized to ensure compatibility with cosine similarity. A FAISS index was constructed over the normalized matrix to support efficient retrieval.

This space models sonic similarity exclusively. Songs that share comparable rhythmic structure perceived energy harmonic brightness and production characteristics tend to appear as nearest neighbors. Qualitative inspection revealed that audio similarity often corresponded closely to genre boundaries since many genres exhibit distinct sonic signatures.

## 4 The Hybrid Similarity Model

The hybrid space integrates both modalities through a weighted linear combination of lyrics and audio cosine similarities. For a seed track \(i\) and candidate track \(j\) the hybrid score is defined as

\[
s_{\text{hybrid}}(i j) = \alpha \cdot \cos(a_i a_j) + (1 - \alpha) \cdot \cos(l_i l_j)
\]

where \(a\) denotes normalized audio embeddings \(l\) denotes normalized lyric embeddings and \(\alpha\) controls the balance between meaning and sound. Retrieval in the hybrid space does not rely on an external index but instead computes hybrid similarities directly from the two matrices.

By adjusting \(\alpha\) the system can emphasize sonic similarity lyrical meaning or a structured blending of both. This property makes the hybrid model a flexible tool for exploring multimodal relationships and understanding how different features influence perceived similarity.

## 5 Evaluation Methodology

Three evaluation metrics were selected to capture complementary aspects of retrieval quality Genre Purity K Emotion Purity K and Artist Diversity K. The goal was to measure the degree to which neighbors reflect consistent genre or emotional labels while also assessing whether the system avoids recommending overly homogeneous sets of artists.

For each metric a set of approximately two hundred seed tracks was sampled from the processed dataset. For each seed its top K neighbors were retrieved in the lyrics audio and hybrid spaces. The metrics were computed as follows

**Genre Purity K** measures the proportion of recommended tracks sharing the same genre as the seed thereby capturing alignment with sonic and stylistic categories.

**Emotion Purity K** evaluates whether neighbors share the same emotion label as the seed track thereby measuring affective alignment and narrative coherence.

**Artist Diversity K** quantifies the proportion of unique artists among the top K neighbors thereby assessing the system’s ability to promote variety instead of clustering excessively around a single artist or album.

All metrics were averaged across the set of seed tracks to compare the three spaces.

## 6 Quantitative Findings

The lyrics semantic space exhibited strong performance on Emotion Purity K. This indicates that semantic embeddings effectively capture affective and thematic qualities allowing the system to retrieve tracks bound by similar emotional tone even when their genres differ. However this space demonstrated lower Genre Purity reflecting the fact that songs with similar lyrical themes may differ significantly in sonic attributes or stylistic conventions. Artist diversity tended to be lower due to strong clustering around artists or songwriting teams with consistent thematic signatures.

The audio feature space produced the highest Genre Purity K across most seeds. This space naturally favors stylistically coherent results as energy rhythmic structure harmonic content and production qualities strongly influence perceived similarity. Nevertheless audio only retrieval did not reliably group songs by emotional content. While sonically coherent these recommendations occasionally lacked narrative or thematic alignment. Artist diversity was moderate to high indicating that sonic similarity is less prone to clustering around specific artists compared to semantic similarity.

The hybrid space demonstrated the most balanced profile. With \(\alpha = 0.4\) hybrid recommendations achieved Genre Purity values that approached or exceeded those of the audio space while simultaneously maintaining significantly higher Emotion Purity. Artist Diversity K remained competitive or superior to that of both unimodal spaces. Hybrid retrieval avoided failure cases observed in the individual spaces it improved narrative coherence relative to audio only retrieval and improved stylistic coherence and diversity relative to lyrics only retrieval.

These findings support the conclusion that multimodal similarity captures a richer and more human aligned representation of musical relationships.

## 7 Impact of Lyrical Features on Recommendation Performance

Introducing lyrical semantic features changed the underlying structure of the recommendation space in ways that cannot be achieved through audio descriptors alone. Lyrics introduced information about narrative intent emotional viewpoint and thematic relationships which expanded the model’s capacity to identify forms of similarity that are independent of production choices or stylistic categories. These features allowed the system to recognize songs that share expressive or conceptual meaning even when their sonic profiles diverge. This shift created a richer representation of musical relatedness and revealed that meaningful similarity is not limited to shared rhythmic or timbral traits. When combined with audio signals the lyrical features provided a second axis of interpretation allowing the hybrid model to balance expressive coherence with stylistic expectation. This interaction produced a recommendation space that reflects more genuinely human patterns of musical understanding by representing both what a track communicates and how it sounds.


## 8 Qualitative Observations

Manual inspection of several representative seed tracks provided further validation of the quantitative trends. Lyrics space neighbors exhibited strong emotional and thematic alignment but sometimes crossed genre boundaries in ways that would not align with listener expectations for playlisting. Audio space neighbors exhibited strong genre coherence but sometimes failed to respect narrative or expressive features encoded in the lyrics.

Hybrid space neighbors generally exhibited both thematic and sonic coherence. Tracks tended to share emotional tone production style rhythmic character and overall expressive quality. In cases where lyrics and audio characteristics diverged the hybrid system tended to favor candidates offering the most balanced combination of both modalities.

## 9 Conclusion

The contextual hybrid model presents a meaningful advancement over unimodal lyric or audio similarity systems. It demonstrates improved robustness interpretability and alignment with musical perception by integrating complementary information channels. Lyrics alone reveal narrative and emotional relationships while audio features capture stylistic rhythmic and energetic structure. The hybrid model unifies these representational strengths producing similarity judgments that better approximate how listeners perceive relationships between pieces of music.

This multimodal architecture therefore provides a foundation for future developments in interpretable recommendation systems including adversarial robustness studies semantic drift analysis and fairness aware evaluation. The results of this analysis confirm the value of integrating contextual and sonic features when modeling similarity in modern recommendation pipelines.

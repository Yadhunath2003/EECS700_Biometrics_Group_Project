# EECS700_Biometrics_Group_Project

### Model Choice

**Option 1**: ArcFace (InsightFace, buffalo_l), our main model

- Widely used, strong face recognition model that cleanly separates identities.
- Easy to set up: one pip package handles detection, alignment, and the 512-D embedding.
- Its face detector copes reasonably well with profile faces.

**Option 2:** AdaFace (IR-101), our comparison model

- Built to handle degraded images, so it may do better on our noisy and blurred probes.
- More setup work: clone the GitHub repo, download the weights, align faces ourselves.

**Plan**

- Run both with the same preprocessing and binarization.
- The model with the higher d′ becomes our final system for Q2–Q7.
- The other appears in Q1 as "alternative tested," in a small comparison table.
- Get ArcFace working end to end first, then add AdaFace.

**Binarization (for both)*

- Normalize the embedding, project it to more dimensions with a fixed random matrix, and set each bit to 1 if the value is above 0, otherwise 0.
- Compare templates with normalized Hamming distance.


### Extra-credit flow: concatenating bit strings

Each image (gallery AND probe)
   │
   ├─► ArcFace ─► 512-D embedding ─► binarize ─► bits A (e.g. 2048 bits)
   │
   └─► AdaFace ─► 512-D embedding ─► binarize ─► bits B (e.g. 2048 bits)
                                                     │
                     Template = [ bits A | bits B ]  (4096 bits)
                                                     │
        Hamming distance on the full template = A's distance + B's distance
                                                     │
        New 994 × 994 score matrix for the weakest condition (from Q7)
                                                     │
        Revised d′  →  Δd′ = revised d′ − baseline d′  (need Δd′ > 1.5)
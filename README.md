# EECS700_Biometrics_Group_Project

### Model Choice

**Option 1**: ArcFace (InsightFace, buffalo_l) or our main model

- It's widely used for strong face recognition model that cleanly separates identities.
- It's easy to set up: one pip package handles detection, alignment, and the 512-D embedding.
- Its face detector copes reasonably well with profile faces.

**Option 2:** AdaFace (IR-101) or our comparison model

- It's built to handle degraded images, so it may do better on our noisy and blurred probe images.
- There is more setup work: clone the GitHub repo, download the weights, and align the images' faces.

**Plan**

- ArcFace is generating binary templates for 41.8 minutes and currently providing the d'value = 7.18999 since the histogram is properly separating the scores' distrubutions.
- Try AdaFace to see if its final results > ArcFace's current results.
- Run these 2 models with the same preprocessing and binarization techniques.
- The model with the high d′ becomes our final system for Q2–Q7.
- Get ArcFace working end to end first. Then, add AdaFace.

**Binarization (for both)**

- Normalize the embedding, project it to more dimensions with a fixed random matrix, and set each bit to 1 if the embedding value >= 0. If not, then set the bit to 0.
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
        Revised d′ →  Δd′ = revised d′ − baseline d′  (need Δd′ > 1.5)

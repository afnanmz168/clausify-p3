# Speech, Slides 1–7 (short, bullet points)

**Speaker: Afnan · about 6 minutes of speaking, around 7 with pauses and clicks · one bullet = one sentence to say · bold = the key word or number to remember · *[brackets]* = what to do, not what to say**

Every number here is taken from the thesis report as submitted. Where the slide wording and the report disagree, the speech follows the report.

---

## Slide 1 · Title · 0:42

- Good morning, respected members of the panel.
- We are group **P25301051**.
- I am Afnan Mazumdar, with Jerin Aktar and Shoyeb Hasan Sayem.
- Our supervisor is **Utsho** Kumar Roy sir.
- We built a **three-model pipeline** on the CUAD dataset.
- It **finds** the important clauses in a contract, **extracts** their text, gives each a **risk level**, and adds a **plain-English** sentence.
- Everything is tested on contracts the models **never saw**.
- **Three of our five research questions came back "no."**
- We present those negative results as our **contribution**, because each one corrects what the component scores alone would suggest.

*[click]*

---

## Slide 2 · The problem, and the data · 1:02

- Contracts are **long**, and the clauses that matter are **short**.
- The median contract is about **33,000 characters**.
- The median clause is only **196 characters**.
- So a contract is about **sixteen times** longer than what a transformer can read at once.
- CUAD has **510 real contracts**, with lawyers marking clauses in **41 categories**.

*[point to the cards]*

- We split it **80/20 at the contract level**, with seed 42.
- That gives **408 contracts to train** and **102 to test**.
- All three models use the **same split**, so no training clause can reach the test set.
- The data is **severely imbalanced**, from 100 percent down to 2.5 percent.
- **Five categories** with seven or fewer test positives score **zero**.
- And the data **cannot be grown**: annotation cost about **two million dollars**.
- So **transfer learning** is mandatory, and every result is on **held-out** contracts.

*[click]*

---

## Slide 3 · The pipeline, and how it reads a contract · 1:08

- Our pipeline has **four stages**.
- **Presence**: TF-IDF and a windowed DistilBERT decide which clauses are there.
- **Span**: DistilBERT-QA extracts the clause text.
- **Risk**: a per-category lookup we wrote from the **weaker party's** side.
- **Explanation**: FLAN-T5 adds a one-line sentence.

*[point to the red and green boxes]*

- Measured end to end, presence loses **22 percent** of clauses.
- Span loses another **56.3 percent**.
- Risk loses **nothing**, because it is a lookup.
- The explanation picks the **wrong template 30 percent** of the time.

*[point to the window diagram]*

- We slide a **2,000-character window** in **1,500-character steps**.
- The **500-character overlap** is longer than the median clause.
- We score every window and take the **maximum**.
- If **any window fires**, the clause is present.
- This **removes the recall ceiling**.
- But it **costs precision**, because one wrong window flips the whole contract.
- Retrieval could find the right window only **64 percent** of the time, below the baseline.
- So we **dropped retrieval** before training.

*[click]*

---

## Slide 4 · How we trained, and how we test · 0:50

- Everything trained on one **Apple M4 laptop** with **16 gigabytes**, no CUDA.

*[point to the rows]*

- Presence took about **49 minutes**, span **23**, and explanation **8**.
- DeBERTa gave **NaN gradients** on the Mac, so we used **DistilBERT**.

*[point to the cards]*

- **No hyperparameter was ever tuned.**
- Only presence had a small validation set: **5 percent of training windows**, for monitoring.
- That **protects the test set**, but the transformer competed **untuned**.
- We report **one mistake**: an early summarizer run monitored loss on **300 test clauses**.
- It selected **no checkpoint**, and the reported model was **rebuilt without it**.
- We test on **102 unseen contracts**, **4,182 decisions** and **2,667 span windows**.
- Our bootstrap resamples **whole contracts, not cells**.

*[click]*

---

## Slide 5 · RQ1: the transformer loses to a bag of words · 0:48

- Our first question: can a windowed transformer beat a full-document baseline?
- The answer is **no**.

*[point to the bar chart]*

- The **TF-IDF baseline** scores **0.775**.
- The **transformer** scores **0.694**.
- The baseline wins by **0.081**, in **every one** of ten thousand bootstrap resamples.
- TF-IDF reads the **whole contract**, and max-pooling creates **false positives**.
- The **AND ensemble** scores **0.776**, but we **withdrew** that win.
- Its lead is only **0.0014**, and its confidence interval **crosses zero**.
- So it is a **tie**.

*[point to the small chart]*

- Formulaic categories are nearly solved: **Document Name 1.000, Parties 0.995, Governing Law 0.961**.
- Rare categories score **zero**.
- That is why the **baseline's macro-F1** is only **0.572**.

*[click]*

---

## Slide 6 · We ranked a leaderboard by noise · 0:52

- We tested **every excuse** for the transformer.

*[point to the bullets]*

- **Undertrained?** Training on all 53,662 windows changed the score by **minus 0.0004**.
- **Unlucky seed?** Across three seeds, the gap is **9.5 standard deviations**.
- **Wrong pooling?** We tried six rules, and **none reaches the baseline**.
- **Bad threshold?** It explains **0.056**, a real caveat, but still not enough.
- So **the transformer loses**.
- It is **not** a budget, seed or aggregation artefact.

*[point to the chart]*

- The ensemble's lead is only **0.0014**.
- Seed noise is **0.0022**.
- Split variance is **0.0063**.
- The bootstrap half-width is **0.0078**.
- **Every error bar is bigger than the result.**
- We keep the ensemble for its **balanced precision and recall**.
- It is also **four times less sensitive** to the random seed.

*[click]*

---

## Slide 7 · The threshold, and what the confidence means · 0:57

- Two things we measured **after the fact**.

*[point to the left plot]*

- **First, the threshold.**
- **0.5** is just the softmax default.
- At **0.9**, the transformer reaches **0.750** instead of **0.694**.
- But we **do not adopt it**.
- Choosing a threshold on **test data** is exactly the tuning our protocol **forbids**.

*[point to the right plot]*

- **Second, the confidence.**
- The calibration error is **0.201**.
- A score above 0.9 means the clause is there only about **73 percent** of the time.
- So our app's confidence bar **overstates certainty**.
- The model also **over-fires**: precision is only **0.562**.
- A maximum over about **35 windows** is **35 chances to be wrong**.

*[point to the green card]*

- The fix: **carve a validation split** from the training contracts.
- Choose the threshold there, and **calibrate** the score there.
- Now **[teammate's name]** will show what these errors mean for the high-risk clauses.

*[hand over]*

---

## Three things the speech says differently from the slides

- **Slide 1:** say "Utsho Kumar Roy" (the slide spells it "Utsha").
- **Slide 5:** say "the **baseline's** macro-F1 is 0.572". The slide puts 0.572 next to the ensemble's 0.776, but it belongs to the TF-IDF baseline (report §5.3).
- **Slide 7:** say "about 35 windows". The slide prints "-35".

---

## Backup answer: "Does the presence model read the whole 2,000-character window?"

Only use this if someone asks. The code truncates each (question, window) input at 256 tokens with no stride, so the model reads roughly the **first 1,000 characters** of each window.

- **No.** At 256 tokens, it reads about the **first half** of each window.
- That is **not discussed** in the report.
- It is likely **one more reason** the transformer loses to TF-IDF, which reads the whole contract.
- None of our four controls changes the windowing, so they **can't rule it out**.
- The fix is cheap: **~900-character windows**, with an overlap still longer than the median clause.

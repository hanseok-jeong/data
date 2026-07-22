# -*- coding: utf-8 -*-
"""
================================================================================
 2_topic_selection.py  :  Gyeongan Stream news text mining - [Step 2] Topic-number selection
================================================================================
 Run order : (1) Preprocessing  ->  (2) Topic-number selection  ->  (3) LDA analysis

 What this script does
   - Load the dictionary and corpus saved in Step 1
   - Vary the number of topics (num_topics) and compute:
       * Perplexity        (lower is better)
       * Coherence u_mass  (same metric as the original)
       * Coherence c_v     (0-1, higher is better / commonly used to pick topic count)
   - Save the scores to CSV so you can inspect them and pick the best topic count
     (in this analysis the result led to choosing 4 topics -> used in Step 3)

 Outputs (in the "outputs" folder)
   - topic_selection_scores.csv  : metric values per topic count (for manual selection)
   - topic_selection_plot.png    : metric curves per topic count
================================================================================
"""

import os
import json

import pandas as pd
import matplotlib.pyplot as plt
from gensim.corpora.dictionary import Dictionary
from gensim.corpora import MmCorpus
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

# ============================================================================
#  CONFIG
# ============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TOPIC_MIN   = 1      # first topic count to test
TOPIC_MAX   = 30     # last topic count to test (inclusive)
PASSES      = 15     # training passes
ITERATIONS  = 400    # iterations
RANDOM_STATE = 7     # fixed for reproducibility


# ============================================================================
#  1. Load Step 1 artifacts
# ============================================================================
dictionary = Dictionary.load(os.path.join(OUTPUT_DIR, "dictionary.dict"))
corpus = list(MmCorpus(os.path.join(OUTPUT_DIR, "corpus.mm")))
with open(os.path.join(OUTPUT_DIR, "texts.json"), encoding="utf-8") as f:
    texts = json.load(f)   # needed for c_v coherence
print("Loaded : dictionary %d words / %d documents" % (len(dictionary), len(corpus)))


# ============================================================================
#  2. Compute metrics per topic count
# ============================================================================
rows = []
for num_topics in range(TOPIC_MIN, TOPIC_MAX + 1):
    ldamodel = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=PASSES,
        iterations=ITERATIONS,
        random_state=RANDOM_STATE,
    )

    perplexity = ldamodel.log_perplexity(corpus)
    cm_umass = CoherenceModel(model=ldamodel, corpus=corpus, coherence="u_mass")
    cm_cv = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence="c_v")
    coh_umass = cm_umass.get_coherence()
    coh_cv = cm_cv.get_coherence()

    rows.append({
        "num_topics": num_topics,
        "perplexity": perplexity,
        "coherence_umass": coh_umass,
        "coherence_cv": coh_cv,
    })
    print("  topics=%2d | perplexity=%8.4f | u_mass=%7.4f | c_v=%6.4f"
          % (num_topics, perplexity, coh_umass, coh_cv))


# ============================================================================
#  3. Save the scores to CSV
# ============================================================================
scores = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR, "topic_selection_scores.csv")
scores.to_csv(csv_path, index=False, encoding="utf-8-sig")
print("\nScores saved :", csv_path)


# ============================================================================
#  4. Save the plot (inspect it to pick the topic count)
# ============================================================================
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel("Number of Topics")
ax1.set_ylabel("Coherence")
ax1.plot(scores.num_topics, scores.coherence_cv, "o-", label="Coherence c_v")
ax1.plot(scores.num_topics, scores.coherence_umass, "s--", label="Coherence u_mass")
ax1.set_xticks(scores.num_topics)

ax2 = ax1.twinx()
ax2.set_ylabel("Perplexity")
ax2.plot(scores.num_topics, scores.perplexity, "^:", color="gray", label="Perplexity")

lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], loc="best")
plt.title("Topic Number Selection")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "topic_selection_plot.png"))
plt.close(fig)
print("Plot saved : topic_selection_plot.png")

print("\n=== Step 2 done -> pick a topic count from the CSV/plot, then run 3_LDA_analysis.py ===")

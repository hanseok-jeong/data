# -*- coding: utf-8 -*-
"""
================================================================================
 3_LDA_analysis.py  :  Gyeongan Stream news text mining - [Step 3] LDA analysis (4 topics)
================================================================================
 Run order : (1) Preprocessing  ->  (2) Topic-number selection  ->  (3) LDA analysis

 Based on Step 2 results, the topic count was chosen as 4 -> NUM_TOPICS = 4

 What this script does
   [A] Static LDA (4 topics)
       - Extract representative keywords per topic -> save CSV
       - Save pyLDAvis visualization as HTML
       - Save the model (lda.model)
   [B] Dynamic topic model LdaSeq (4 topics, the paper's trend figure)
       - Topic weight trend by year -> save CSV
       - Topic keyword trend by year -> save CSV
       - Save the trend plot (dtm.png)
       - Save the model (ladseq_4topic.model)

 Outputs (in the "outputs" folder)
   - lda_topics.csv                : static LDA keywords per topic
   - lda_vis.html                  : pyLDAvis visualization
   - lda.model                     : static LDA model
   - topic_trend_mean_values.csv   : topic weight trend by year (paper figure data)
   - topic_trend_keywords.csv      : topic keyword trend by year
   - dtm.png                       : topic trend plot by year
   - ladseq_4topic.model           : dynamic topic model
================================================================================
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.corpora.dictionary import Dictionary
from gensim.corpora import MmCorpus
from gensim.models.ldamodel import LdaModel
from gensim.models import LdaSeqModel

warnings.filterwarnings("ignore")

# ============================================================================
#  CONFIG
# ============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

NUM_TOPICS   = 4      # topic count chosen in Step 2
TOP_N_WORDS  = 20     # number of representative keywords per topic
RANDOM_STATE = 7


# ============================================================================
#  1. Load Step 1 artifacts
# ============================================================================
dictionary = Dictionary.load(os.path.join(OUTPUT_DIR, "dictionary.dict"))
corpus = list(MmCorpus(os.path.join(OUTPUT_DIR, "corpus.mm")))
with open(os.path.join(OUTPUT_DIR, "meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
time_slice = meta["time_slice"]   # document count per year
time_tag = meta["time_tag"]       # year labels
print("Loaded : dictionary %d words / %d documents / %d years"
      % (len(dictionary), len(corpus), len(time_tag)))


# ============================================================================
#  [A] Static LDA (4 topics)
# ============================================================================
print("\n[A] Training static LDA ...")
lda = LdaModel(
    corpus=corpus, id2word=dictionary,
    num_topics=NUM_TOPICS, chunksize=1000, random_state=RANDOM_STATE,
)

# Keywords per topic -> CSV
topics = lda.print_topics(num_words=50)
topics_df = pd.DataFrame(topics, columns=["topic", "keywords"])
topics_df.to_csv(os.path.join(OUTPUT_DIR, "lda_topics.csv"), index=False, encoding="utf-8-sig")
lda.save(os.path.join(OUTPUT_DIR, "lda.model"))
print("    Topic keywords saved : lda_topics.csv / model saved : lda.model")

# pyLDAvis visualization (only if installed)
try:
    import pyLDAvis
    try:
        import pyLDAvis.gensim_models as gensimvis   # newer versions
    except ImportError:
        import pyLDAvis.gensim as gensimvis          # older versions
    vis = gensimvis.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(vis, os.path.join(OUTPUT_DIR, "lda_vis.html"))
    print("    pyLDAvis saved : lda_vis.html")
except Exception as e:
    print("    (skipped) pyLDAvis visualization failed:", e)


# ============================================================================
#  [B] Dynamic topic model LdaSeq (4 topics, the paper's trend)
# ============================================================================
print("\n[B] Training dynamic topic model (LdaSeq) ... (may take a while)")
ldaseq = LdaSeqModel(
    corpus=corpus, id2word=dictionary,
    time_slice=time_slice, num_topics=NUM_TOPICS, random_state=RANDOM_STATE,
)
ldaseq.save(os.path.join(OUTPUT_DIR, "ladseq_4topic.model"))


def get_topic_words(topic, top_n_words=TOP_N_WORDS):
    words, _ = zip(*topic)
    return list(words[:top_n_words])


# (B-1) Topic keyword trend by year -> save as a single CSV
rows = []
for topic_id in range(NUM_TOPICS):
    topic_times = ldaseq.print_topic_times(topic_id)   # topic words per time slice
    for t_idx, topic in enumerate(topic_times):
        words = get_topic_words(topic, TOP_N_WORDS)
        rows.append({
            "topic": topic_id,
            "year": time_tag[t_idx],
            "keywords": ", ".join(words),
        })
pd.DataFrame(rows).to_csv(
    os.path.join(OUTPUT_DIR, "topic_trend_keywords.csv"),
    index=False, encoding="utf-8-sig",
)
print("    Topic keyword trend saved : topic_trend_keywords.csv")


# (B-2) Topic weight trend by year -> CSV (paper figure data)
def get_topic_trends(model, corpus, time_slice):
    dtm = model.dtm_vis(0, corpus)
    trends, start = [], 0
    for index in time_slice:
        trends.append(np.array(dtm[0][start:start + index]).mean(axis=0))
        start += index
    return np.array(trends)


topic_trends = get_topic_trends(ldaseq, corpus, time_slice)
trend_df = pd.DataFrame(
    topic_trends,
    columns=["Topic %d" % (i + 1) for i in range(NUM_TOPICS)],
)
trend_df.insert(0, "Year", time_tag)
trend_df.to_csv(
    os.path.join(OUTPUT_DIR, "topic_trend_mean_values.csv"),
    index=False, encoding="utf-8-sig",
)
print("    Topic weight trend saved : topic_trend_mean_values.csv")


# (B-3) Trend plot
topic_titles = ["Topic " + str(i + 1) for i in range(NUM_TOPICS)]
fig, axes = plt.subplots(2, 2, sharex="col", figsize=(12, 6))
for i, (title, ax) in enumerate(zip(topic_titles, axes.ravel())):
    ax.set_title(title)
    ax.plot(topic_trends[:, i])
    ax.set_xticks(range(0, len(time_tag), 4))
    ax.set_xticklabels(time_tag[::4])
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "dtm.png"))
plt.close(fig)
print("    Trend plot saved : dtm.png")

print("\n=== Step 3 LDA analysis done -> check the results in the outputs folder ===")

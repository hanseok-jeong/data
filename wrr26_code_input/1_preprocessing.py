# -*- coding: utf-8 -*-
"""
================================================================================
 1_preprocessing.py  :  Gyeongan Stream news text mining - [Step 1] Preprocessing
================================================================================
 Run order : (1) Preprocessing  ->  (2) Topic-number selection  ->  (3) LDA analysis

 What this script does
   1) Load news CSV (Date, content) and derive year (yearmo)
   2) Regex cleaning -> Mecab noun extraction -> stopword removal
   3) Build and save gensim Dictionary + corpus
      -> reused by Steps 2 and 3 (no need to preprocess again)
   4) [Added] Save a word cloud image
   5) [Added] Save TF-IDF values for the top 10 keywords as CSV

 Outputs (in the "outputs" folder)
   (also prints the vocabulary size before/after filtering, and the corpus size)

   - dictionary.dict        : gensim dictionary
   - corpus.mm              : gensim corpus
   - texts.json             : noun tokens per document (for Step 2/3 coherence & LdaSeq)
   - meta.json              : time_slice, time_tag (for the Step 3 dynamic model)
   - wordcloud.png          : word cloud
   - tfidf_top10.csv        : top 10 keywords by TF-IDF value
================================================================================
"""

import os
import re
import json

import pandas as pd
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora.dictionary import Dictionary
from gensim.corpora import MmCorpus
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ============================================================================
#  CONFIG  -- edit these paths for your environment
# ============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV   = os.path.join(BASE_DIR, "NewsResult_kyan1.csv")   # input news data
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")               # output folder
MECAB_DIC   = r"C:\mecab\mecab-ko-dic"                        # Mecab Korean dictionary
STOPWORD_TXT = r"D:\news\stopword.txt"                        # stopword file (skipped if missing)
WC_FONT     = r"C:/Windows/Fonts/malgun.ttf"                  # Korean font for word cloud

# Dictionary filter options (same as original)
FILTER_KEEP_N   = 9600   # keep only the top-N most frequent tokens
FILTER_NO_BELOW = 5      # keep words appearing at least this many times
FILTER_NO_ABOVE = 0.5    # drop words appearing in more than this fraction of documents

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
#  1. Load data + derive year (yearmo)
# ============================================================================
df = pd.read_csv(INPUT_CSV, encoding="CP949")

d_year = pd.DatetimeIndex(df.Date).strftime("%Y")
d_y = pd.DataFrame(d_year, columns=["yearmo"])
result = pd.concat([df, d_y], axis=1)
result = result.sort_values(by=["yearmo"], ascending=True).reset_index(drop=True)

# Time information used by the dynamic topic model (Step 3)
time_slice = list(result["yearmo"].value_counts().sort_index())  # document count per year
time_tag = sorted(list(set(result["yearmo"])))                    # year labels
print("[1/5] Data loaded :", len(result), "articles /", len(time_tag), "years")


# ============================================================================
#  2. Regex cleaning
# ============================================================================
content = result.content
content = content.str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣0-8a-z]", " ", regex=True)  # remove symbols/digits
content = content.replace(r"^\s+|\s+$", "", regex=True)                      # trim leading/trailing spaces
content = content.replace("[一-龥]", "", regex=True)                         # remove Chinese characters
content = content.replace(" +", " ", regex=True)                            # collapse multiple spaces
content = content.replace("도시 숲", "도시숲", regex=True)                   # keep compound word


# ============================================================================
#  3. Mecab noun extraction + stopword removal
# ============================================================================
mecab = Mecab(dicpath=MECAB_DIC)

# Load stopwords (proceed with empty list if file is missing)
try:
    stop_words = pd.read_csv(STOPWORD_TXT, header=None)
    stopword = stop_words[0].values.tolist()
except FileNotFoundError:
    print("    (warning) stopword file not found, skipping stopword removal:", STOPWORD_TXT)
    stopword = []


def combine_continuous_words(word_list):
    """Merge consecutive tokens such as '지속' + '가능' into '지속가능'."""
    result_words = []
    skip_next = False
    for i in range(len(word_list)):
        if skip_next:
            skip_next = False
            continue
        if word_list[i] == "지속" and i + 1 < len(word_list) and word_list[i + 1] == "가능":
            result_words.append("지속가능")
            skip_next = True
        else:
            result_words.append(word_list[i])
    return result_words


doc_nouns_list = []          # noun tokens per document (list of lists)
for doc in content:
    nouns = mecab.nouns(doc)
    combined = combine_continuous_words(nouns)
    nouns_list = [w for w in combined if w not in stopword]
    nouns_list = [n for n in nouns_list if len(n) > 1]   # drop single-character tokens
    doc_nouns_list.append(nouns_list)
print("[2/5] Noun extraction done")


# ============================================================================
#  4. Build/save gensim dictionary + corpus
# ============================================================================
dictionary = Dictionary(doc_nouns_list)
vocab_before = len(dictionary)   # vocabulary right after preprocessing (before filtering)
dictionary.filter_extremes(keep_n=FILTER_KEEP_N, no_below=FILTER_NO_BELOW, no_above=FILTER_NO_ABOVE)
vocab_after = len(dictionary)    # final vocabulary actually fed into LDA
corpus = [dictionary.doc2bow(text) for text in doc_nouns_list]

dictionary.save(os.path.join(OUTPUT_DIR, "dictionary.dict"))
MmCorpus.serialize(os.path.join(OUTPUT_DIR, "corpus.mm"), corpus)
with open(os.path.join(OUTPUT_DIR, "texts.json"), "w", encoding="utf-8") as f:
    json.dump(doc_nouns_list, f, ensure_ascii=False)
with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump({"time_slice": time_slice, "time_tag": time_tag}, f, ensure_ascii=False)
print("[3/5] Saved dictionary and corpus")
print("      Vocabulary after preprocessing           : %d" % vocab_before)
print("      Vocabulary after filter_extremes (final) : %d" % vocab_after)
print("      Documents (corpus size)                  : %d" % len(corpus))


# ============================================================================
#  5-A. [Added] Save word cloud
# ============================================================================
word_tf_dic = {}
for tokens in doc_nouns_list:
    for w in tokens:
        word_tf_dic[w] = word_tf_dic.get(w, 0) + 1

wordcloud = WordCloud(
    font_path=WC_FONT,
    background_color="white",
    colormap="Accent_r",
    width=800, height=800, max_words=200,
)
wordcloud.generate_from_frequencies(word_tf_dic)

fig = plt.figure(figsize=(10, 10))
plt.imshow(wordcloud.to_array(), interpolation="bilinear")
plt.axis("off")
fig.savefig(os.path.join(OUTPUT_DIR, "wordcloud.png"), bbox_inches="tight")
plt.close(fig)
print("[4/5] Word cloud saved : wordcloud.png")


# ============================================================================
#  5-B. [Added] Save TF-IDF values for the top 10 keywords
# ============================================================================
joined_docs = [" ".join(tokens) for tokens in doc_nouns_list]     # join each document into a string
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(joined_docs)

terms = tfidf.get_feature_names_out()
mean_tfidf = tfidf_matrix.mean(axis=0).A1                         # mean TF-IDF per word
tfidf_df = (
    pd.DataFrame({"keyword": terms, "tfidf": mean_tfidf})
    .sort_values("tfidf", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
tfidf_df.to_csv(os.path.join(OUTPUT_DIR, "tfidf_top10.csv"), index=False, encoding="utf-8-sig")
print("[5/5] Top 10 TF-IDF keywords saved : tfidf_top10.csv")
print(tfidf_df)

print("\n=== Step 1 preprocessing done -> check the outputs folder, then run 2_topic_selection.py ===")

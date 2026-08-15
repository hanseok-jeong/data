# Reproducibility Materials

This repository contains the data and Python scripts used for the computational text analysis reported in:

Hydrology Meets Public Attention and Policy Change: Integrating a Multiple Streams Framework and Human Social Sensing into Sociohydrological Analysis

The scripts reproduce the news-data processing and topic-modeling workflow used to derive the Human Social Sensing (HSS) indicators of media attention described in the manuscript and Supporting Information.

## Repository structure

wrr2026_reproducibility/
├── input_data/
├── 0_crawling_newspaper.py
├── 1_preprocessing.py
├── 2_topic_selection.py
├── 3_LDA_analysis.py
└── README.md

input_data/

Contains the input and intermediate data required by the analysis scripts. The materials support the computational text-analysis workflow described in Sections 2.4–2.4.2 of the manuscript and Texts S2–S4 of the Supporting Information.

0_crawling_newspaper.py

Collects environmental news articles from the sources used in the study and stores article-level information for subsequent processing. The study compiled articles from Naver News and BigKinds for 1997–2021 and retained records relevant to the Kyung-An Stream after duplicate detection and relevance screening.

1_preprocessing.py

Preprocesses the collected Korean-language news corpus for text analysis. The preprocessing workflow includes text cleaning, morphological analysis, noun extraction, vocabulary filtering, and preparation of the corpus used for TF–IDF and topic modeling.

2_topic_selection.py

Evaluates candidate Latent Dirichlet Allocation (LDA) topic models using coherence and perplexity. Candidate models with different numbers of topics are compared, and the diagnostics are used to identify the four-topic solution adopted in the study.

3_LDA_analysis.py

Runs the final four-topic LDA model and generates the topic-level outputs used in the manuscript. The resulting topic proportions are aggregated by year to construct the annual, topic-specific media-attention indicators used in the HSS analysis.

## Recommended execution order

Run the scripts in the following order:

0_crawling_newspaper.py
        ↓
1_preprocessing.py
        ↓
2_topic_selection.py
        ↓
3_LDA_analysis.py

In brief:

Data collection — 0_crawling_newspaper.py

Text preprocessing — 1_preprocessing.py

Topic-number selection — 2_topic_selection.py

Final LDA analysis and HSS indicator generation — 3_LDA_analysis.py

## Methodological correspondence

The computational workflow corresponds to the following parts of the accompanying publication:

Digital news data collection: Manuscript Section 2.4.1; Supporting Information Text S2

Text preprocessing and TF–IDF: Manuscript Section 2.4.2; Supporting Information Text S3

Topic-model selection: Manuscript Section 2.4.2; Supporting Information Text S4 and Figure S3

Final four-topic LDA analysis: Manuscript Sections 2.4.2 and 3.2; Supporting Information Table S5

Annual HSS indicator: Annual mean document-level topic proportions for each topic, as described in Manuscript Section 2.4

## Notes

The repository is intended to support transparency and reproducibility of the computational text-analysis component of the study. Hydrological modeling procedures, calibration/validation information, and hydrological scenario results are documented separately in the manuscript and Supporting Information.

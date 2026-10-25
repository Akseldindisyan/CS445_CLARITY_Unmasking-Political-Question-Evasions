# CLARITY: Unmasking Political Question Evasions

An end-to-end Natural Language Processing (NLP) and Transformer modeling pipeline designed for **SemEval-2026 Task 6 (CLARITY)**. This project tackles political question evasion detection by classifying question-answer pairs from political interviews into three distinct categories: **Clear Reply**, **Ambivalent**, and **Clear Non-Reply**.

To solve this, we benchmarked two distinct approaches: classical machine learning models driven by extensive linguistic feature engineering, and end-to-end transformer architectures (RoBERTa and Llama 8B/70B) operating directly on raw text. Our top-performing models achieve up to a **0.74 Macro F1 score** on a highly imbalanced political discourse dataset.
---

## Dataset Overview

The system is trained and evaluated on the [`ailsntua/QEvasion`](https://huggingface.co/datasets/ailsntua/QEvasion) benchmark dataset, consisting of manually annotated political interview transcripts from U.S. presidential press conferences:
* **Training Set:** 3,448 question-answer pairs
* **Test Set:** 308 held-out question-answer pairs
* **Target Classes (`clarity_label`):**
  * `Clear Reply` (Direct, explicit answers)
  * `Ambivalent` (Dodging, deflection, general, implicit, or partial/half-answers)
  * `Clear Non-Reply` (Claims ignorance, clarification requests, or declining to answer)

Because the dataset exhibits severe class imbalance—with `Ambivalent` responses dominating the distribution and `Clear Non-Reply` heavily underrepresented—all models are optimized and evaluated using **Macro F1**, **Macro Precision**, and **Macro Recall**.

---

## Methodology & Pipeline Architecture

### 1. Exploratory Data Analysis (EDA) & Preprocessing
* Analyzed response clarity distributions across presidents, interview years, question orders, multi-part questions, affirmative questions, and inaudible transcript flags.
* Built a dual-track preprocessing pipeline using **NLTK** and **spaCy**, accelerated via NLP pipelines and parallel execution:
  * **Feature-Based & Classical Models:** Lowercasing, punctuation removal, lemmatization, stemming, and selective stopword removal that preserves negation cues and personal pronouns critical for evasion detection.
  * **Transformer Models:** Preserved raw contextual text and tokenized paired `(question, answer)` sequences using `RobertaTokenizerFast` (padded and truncated to 256 tokens).

### 2. Linguistic & Statistical Feature Engineering
We engineered **28 handcrafted features** capturing surface-level, syntactic, semantic, and discourse-level evasion signals:
* **Structural & Length Statistics:** Question and answer word counts, sentence counts, punctuation indicators, and answer-to-question length ratios.
* **Lexical & Semantic Alignment:** Jaccard similarity, content-word overlap, named-entity overlap, and contextual cosine similarity via **SBERT** embeddings.
* **Discourse & Pragmatic Markers:** Hedge marker frequencies extracted via **ELFEN**, negation ratios extracted via **spaCy** (`spacy neg`), verb tense distributions, question polarity, and combativeness scores.

### 3. Model Architectures
We benchmarked four tiers of architectures to evaluate the trade-off between handcrafted linguistic features and end-to-end contextual representations:
1. **Classical ML Baselines:** Logistic Regression, Linear SVM, Gaussian Naive Bayes, and Decision Trees trained on TF-IDF representations and handcrafted features.
2. **Advanced ML & Neural Networks:** Random Forest, XGBoost, and a PyTorch Multi-Layer Perceptron (MLP) trained on the 28-dimensional linguistic feature set.
3. **Encoder Transformers (`RoBERTa-base`):** End-to-end sequence-pair classification fine-tuned via the Hugging Face `Trainer` API with FP16 mixed precision and **Optuna** hyperparameter optimization (tuning learning rate, batch size, epochs, and weight decay), evaluated both with and without numeric grounding features.
4. **Decoder Transformers (`Llama-3.1-8B-Instruct` & `Llama-3.3-70B-Instruct`):** Instruction-tuned Large Language Models fine-tuned via **LoRA** (rank 32) on paired question-answer prompts.

---

## Results Summary

Fine-tuning transformer models and neural architectures on the CLARITY benchmark yielded a peak performance of **0.74 Macro F1**, substantially outperforming classical feature-only baselines:

| Model Category | Architecture | Input Representation | Key Highlights |
| :--- | :--- | :--- | :--- |
| **Classical Baselines** | Linear SVM / Logistic Regression / Naive Bayes | TF-IDF + Handcrafted Features | Linear SVM achieved the best efficiency-to-performance trade-off among linear models. |
| **Tree Ensembles** | Random Forest / XGBoost | 28 Handcrafted Features | Captured non-linear feature interactions (0.530 and 0.508 validation Macro F1). |
| **Neural Network** | PyTorch MLP | 28 Handcrafted Features | Improved minority class recall (`Clear Non-Reply` recall of 0.69; 0.564 validation Macro F1). |
| **Encoder & Decoder Transformers** | **RoBERTa-base & Llama 8B / 70B Instruct** | Paired `(Question, Answer)` Sequences + Grounding | Achieved up to **0.74 Macro F1**, effectively capturing subtle semantic shifts and evasive rhetoric. |

---

## Tech Stack & Libraries

* **Deep Learning & Transformers:** PyTorch, Hugging Face `transformers`, `datasets`, `Optuna`, LoRA (Tinker), Sentence-Transformers (SBERT)
* **NLP & Feature Extraction:** spaCy, NLTK, ELFEN
* **Machine Learning & Data Analysis:** Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib

---

## References

1. K. Thomas, G. Filandrianos, M. Lymperaiou, C. Zerva, and G. Stamou, *"I Never Said That": A dataset, taxonomy and baselines on response clarity classification*, Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 5204–5233.

```bibtex
@misc{thomas2024isaidthatdataset,
  title={"I Never Said That": A dataset, taxonomy and baselines on response clarity classification}, 
  author={Konstantinos Thomas and Giorgos Filandrianos and Maria Lymperaiou and Chrysoula Zerva and Giorgos Stamou},
  year={2024},
  eprint={2409.13879},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={[https://arxiv.org/abs/2409.13879](https://arxiv.org/abs/2409.13879)}, 
}

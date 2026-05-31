# Support Ticket Classifier

An NLP classification project that applies TF-IDF vectorization and supervised learning to predict the topic category of customer support tickets from their text descriptions — motivated by a real problem in Knowledge-Centered Service environments where analysts struggle to categorize knowledge articles consistently.

---

## The Business Problem

In KCS-practicing organizations, knowledge articles need to be categorized correctly to be findable. When analysts miscategorize articles, those articles become invisible to the people who need them. A model that can predict the correct category from the text of a ticket or article description would reduce that friction — and this project was a first attempt at building one.

Because no suitable real-world dataset was available, the project uses the [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) from Kaggle as a proxy, with the `Ticket Description` field as features and `Ticket Subject` as the target.

---

## Data Preparation

The dataset required several preprocessing steps before modeling:

- **Template substitution** — ticket descriptions contained `{product_purchased}` placeholders that were replaced with actual product names using regex
- **Text normalization** — lowercasing, punctuation removal, stopword removal via NLTK
- **Stemming** — PorterStemmer applied to reduce words to root forms
- **TF-IDF vectorization** — producing a 6,756-feature matrix from the cleaned descriptions

One notable data quality issue: the dataset appears to have been programmatically generated rather than sourced from real tickets. Descriptions share repetitive phrasing and contain incoherent sentences. This was identified early and factored into the interpretation of results throughout.

---

## Modeling: Four Attempts

The project required four rounds of modeling before arriving at a workable result. Each failed attempt produced a specific diagnostic that informed the next.

### Attempt 1 — 16-class classification
**Model selected:** Decision Tree (max_depth=10) via GridSearchCV  
**Accuracy:** 5.84%  
**Diagnosis:** The model predicted nearly all observations as a single class ("Refund Request"). With 16 target classes and noisy synthetic text, the signal wasn't strong enough to distinguish between categories. Worse than random chance.

### Attempt 2 — Reduce to 5 classes
**Model selected:** Multinomial Naive Bayes  
**Accuracy:** 24.73%  
**Diagnosis:** Collapsing 16 classes to 5 improved accuracy slightly above random chance (20%), but the model still lumped most predictions into a single class ("Other Issue"). The class imbalance problem persisted.

### Attempt 3 — Feature selection (top 500 features)
**Model selected:** Multinomial Naive Bayes  
**Accuracy:** ~23%  
**Diagnosis:** Narrowing to the 500 most informative features via chi-squared selection produced marginally more prediction diversity but no meaningful accuracy improvement. The multi-class problem appeared fundamentally limited by the data quality.

### Attempt 4 — Binary classification with upsampling ✓
**Model selected:** Logistic Regression  
**Accuracy:** 79.25%  
**Approach:** Reframed as a binary problem — predicting whether a ticket belongs to the "Hardware Issue" category. Because Hardware Issue represented only ~20% of observations, the minority class was upsampled to match the majority before training.  
**Result:** A well-balanced confusion matrix with predictions distributed meaningfully across both classes. Substantially better than random chance (50%) and a genuine improvement over all prior attempts.

---

## Key Takeaways

**The multi-class problem remains unsolved for this dataset.** The synthetic data quality appears to be a fundamental ceiling — ticket descriptions that don't reflect real language patterns limit what any text-based classifier can learn. On real support data, multi-class performance would likely be significantly better.

**One path forward for multi-class classification** is to build separate binary classifiers for each category — one "Is this a Hardware Issue?" model, one "Is this a Billing issue?" model, and so on — then combine their outputs to assign a final category. This would extend the approach from Attempt 4 across all classes.

**Upsampling was critical.** Without addressing the class imbalance in Attempt 4, the binary model would have faced the same single-class prediction problem as the earlier attempts.

---

## Stack

- Python, Pandas, NumPy
- NLTK (tokenization, stopwords, stemming)
- Scikit-learn (TF-IDF, GridSearchCV, Logistic Regression, Decision Tree, Multinomial Naive Bayes)
- Matplotlib

---

## Data Source

[Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) — Kaggle. Dataset appears to be synthetically generated; real-world performance on live ticket data may differ significantly from results reported here.

# Artificial Intelligence Coursework

Three applied AI projects built in Python on Google Colab as group coursework for **CN6005 Artificial Intelligence** at the University of East London (2026/27).

| Task | Project | Methods | Key result |
|---|---|---|---|
| 1 | Flower image classification (5 classes) | ANN vs CNN (TensorFlow/Keras), data augmentation, Batch Normalisation, Dropout, Global Average Pooling | CNN 75.49% test accuracy vs ANN 44.42%, with ~41x fewer parameters |
| 2 | Clustering real and fake job adverts | NLTK pre-processing, TF-IDF + LSA, K-Means, Sentence-BERT | TF-IDF beat Sentence-BERT on all cluster quality measures; Sentence-BERT grouped fake adverts more strongly (17.96% vs 4.76% overall) |
| 3 | Travel Information Agent | Sentence-BERT retrieval over a 26-entry knowledge base + Qwen2.5-1.5B-Instruct, city memory | 11/11 in-scope questions answered with the correct information, 3/3 out-of-scope questions declined |

## Repository structure
```
├── Task1_ImageClassification.ipynb   ANN vs CNN on the Flowers "Five Classes" dataset
├── Task2_TextMining.ipynb            TF-IDF, K-Means and Sentence-BERT clustering of job adverts
├── Task3_AIAgent.ipynb               Travel Information Agent (retrieval + LLM)
├── scripts/                          Python script (.py) versions of the three notebooks
├── datasets/links.txt                Dataset sources
├── requirements.txt                  Python libraries
└── README.txt                        How to run the code
```

## How to run
1. Open a notebook in Google Colab (File → Open notebook → GitHub).
2. Choose Runtime → Change runtime type → **T4 GPU**.
3. Run Runtime → Run all. Datasets are downloaded automatically with kagglehub, and the Task 3 models are downloaded from Hugging Face.

To run locally: `pip install -r requirements.txt`, then open the notebooks in Jupyter.

## Datasets
- [Flowers "Five Classes" (Kaggle, lara311)](https://www.kaggle.com/datasets/lara311/flowers-five-classes)
- [Real / Fake Job Posting Prediction (Kaggle, Shivam Bansal)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Task 3 knowledge base: 26 travel entries written by the group (inside `Task3_AIAgent.ipynb`)

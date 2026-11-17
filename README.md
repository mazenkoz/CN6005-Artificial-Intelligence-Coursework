# Artificial Intelligence Coursework

Three applied AI projects written in Python as group coursework for **CN6005 Artificial Intelligence** at the University of East London (2026/27).

| Task | Project | Methods | Key result |
|---|---|---|---|
| 1 | Flower image classification (5 classes) | ANN vs CNN (TensorFlow/Keras), data augmentation, Batch Normalisation, Dropout, Global Average Pooling | CNN 75.49% test accuracy vs ANN 44.42%, with ~41x fewer parameters |
| 2 | Clustering real and fake job adverts | NLTK pre-processing, TF-IDF + LSA, K-Means, Sentence-BERT | TF-IDF beat Sentence-BERT on all cluster quality measures; Sentence-BERT grouped fake adverts more strongly (17.96% vs 4.76% overall) |
| 3 | Travel Information Agent | Sentence-BERT retrieval over a 26-entry knowledge base + Qwen2.5-1.5B-Instruct, city memory | 11/11 in-scope questions answered with the correct information, 3/3 out-of-scope questions declined |

## Repository structure
```
├── Task1_ImageClassification.py   ANN vs CNN on the Flowers "Five Classes" dataset
├── Task2_TextMining.py            TF-IDF, K-Means and Sentence-BERT clustering of job adverts
├── Task3_AIAgent.py               Travel Information Agent (retrieval + LLM)
├── datasets/links.txt             Dataset sources
├── requirements.txt               Python libraries
└── README.txt                     How to run the code
```

## How to run
```bash
pip install -r requirements.txt
python Task1_ImageClassification.py
python Task2_TextMining.py
python Task3_AIAgent.py            # add --chat to talk to the agent live
```
Datasets are downloaded automatically with kagglehub (a free Kaggle account/API token is needed), and the Task 3 models are downloaded from Hugging Face. Figures are saved to a `screenshots/` folder. A GPU is recommended for Tasks 1 and 3.

## Datasets
- [Flowers "Five Classes" (Kaggle, lara311)](https://www.kaggle.com/datasets/lara311/flowers-five-classes)
- [Real / Fake Job Posting Prediction (Kaggle, Shivam Bansal)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Task 3 knowledge base: 26 travel entries written by the group (inside `Task3_AIAgent.py`)

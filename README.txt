CN6005 Artificial Intelligence - Group Coursework
=================================================

Requirements: Python 3.10+ and the libraries in requirements.txt
    pip install -r requirements.txt

How to run
----------
    python Task1_ImageClassification.py
    python Task2_TextMining.py
    python Task3_AIAgent.py          (add --chat to chat with the agent live)

- Tasks 1 and 2 download their Kaggle dataset automatically with kagglehub.
  To use a dataset you already downloaded, set DATA_DIR to its folder (Task 1).
- Task 3 downloads the Sentence-BERT and Qwen2.5-1.5B-Instruct models from Hugging Face (~3 GB, first run only).
- Every figure is saved to a "screenshots" folder next to the scripts; results are printed in the terminal.
- A GPU is recommended for Tasks 1 and 3 (they also run on CPU, just slower).

Files
-----
Task1_ImageClassification.py  ANN vs CNN on the Flowers "Five Classes" dataset (5 classes, 2,746 photos).
                              Includes a baseline CNN to show overfitting and a regularised final CNN.
Task2_TextMining.py           Pre-processing, TF-IDF, K-Means (Elbow + Silhouette) and Sentence-BERT
                              clustering of 5,000 job adverts, compared with five quality measures.
Task3_AIAgent.py              Travel Information Agent: searches a 26-entry knowledge base with
                              Sentence-BERT and uses Qwen2.5-1.5B-Instruct to write the answer.

Notes
-----
- Random seeds are fixed (42) so results can be reproduced.

CN6005 Artificial Intelligence - Group Coursework
=================================================

All notebooks are designed for Google Colab (https://colab.research.google.com).

How to run
----------
1. Open the notebook in Google Colab (File > Upload notebook, or File > Open notebook > GitHub).
2. Set Runtime > Change runtime type > T4 GPU (needed for Tasks 1 and 3, faster for Task 2).
3. Run all cells in order (Runtime > Run all). Each notebook installs/downloads what it needs:
   Tasks 1 and 2 download their Kaggle dataset with kagglehub, and Task 3 downloads the
   Sentence-BERT and Qwen models from Hugging Face. No manual download is needed.
4. Figures and tables are saved to a "screenshots" folder in the Colab session.
   Download it from the Files panel before closing Colab.

To run locally instead:  pip install -r requirements.txt  then open the notebooks in Jupyter.
The scripts/ folder contains .py versions of the notebooks.

Notebooks
---------
Task1_ImageClassification.ipynb  ANN vs CNN on the Flowers "Five Classes" dataset (5 classes, 2,746 photos).
                                 Includes a baseline CNN to show overfitting and a regularised final CNN.
Task2_TextMining.ipynb           Pre-processing, TF-IDF, K-Means (Elbow + Silhouette) and Sentence-BERT
                                 clustering of 5,000 job adverts, compared with five quality measures.
Task3_AIAgent.ipynb              Travel Information Agent: searches a 26-entry knowledge base with
                                 Sentence-BERT and uses Qwen2.5-1.5B-Instruct to write the answer.
                                 Set RUN_CHAT = True in section 3.2.5 to chat with the agent live.

Notes
-----
- Random seeds are fixed (42) so results can be reproduced.
- Task 3 downloads the Qwen model (~3 GB) on first run.

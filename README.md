# AutoQuesGen
Generate questions with multiple choice answers from the user given passage.

# Key Insights

- The proposed approach uses T5 Transformer.
- Keywords are also extracted from the user given passage to frame the questions.
- Sense2Vec is applied to get "distractors".


## Install Anaconda version 4.10.3

1. Create a new environment.(Code: conda create --name env2)
2. Activate env2. (Code: conda activate env2)
3. Install python.(Code: conda install -c anaconda python=3.8)
4. Install spacy. (Code: pip install spacy==3.3.1)
4. Install library. (Code: python -m spacy download en_core_web_sm)
4. Deactivate env2 to go to base environment. (Code: conda deactivate)
5. Now run "test.py" on the terminal.(Code: python test.py)

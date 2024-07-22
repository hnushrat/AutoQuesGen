
import os
import sys
import subprocess
#os.system('conda init --bash')

subprocess.run('/bin/bash -c "source activate env2; python Keyword_extractor.py; pip install numpy"', shell=True)
#subprocess.run('/bin/bash -c "which python"', shell=True)

"""
os.system('Keyword_extractor.py')
os.system('conda activate base')
"""
import numpy as np

keys = np.load('keywords.npy')

print(keys)

os.system('python Ques_Ans.py')




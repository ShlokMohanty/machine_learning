import os 
import pandas as pd 
import torch 
import torchaudio 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
%matplotlib inline
import librosa
import librosa.display
import IPython.display as ipd
import sklearn
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
import plotly.express as px
fig, ax = plt.subplots(6, figsize=(16, 12))
fig.suptitle('Sound Waves', fontsize=16)
librosa.display.waveplot(y=audio_astfly, sr=sr_astfly, color='#A300F9', ax=ax[0])
librosa.display.waveplot(y=audio_casvir, sr=sr_casvir, color="4300FF", ax=ax[1])
librosa.display.waveplot(y = audio_subfly, sr = sr_subfly, color = "#009DFF", ax=ax[2])
librosa.display.waveplot(y = audio_wilfly, sr = sr_wilfly, color = "#00FFB0", ax=ax[3])
librosa.display.waveplot(y = audio_verdin, sr = sr_verdin, color = "#D9FF00", ax=ax[4])
librosa.display.waveplot(y = audio_solsan, sr = sr_solsan, color = "r", ax=ax[5]);
for i, name in zip(range(6), birds):
  ax[i].set_ylabel(name, fontsize=13)

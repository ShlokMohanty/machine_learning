n_fft =  2048
hop_length = 512
D_astfly = np.abs(librosa.stft(audio_astfly, n_fft = n_fft, hop_length=hop_length))
DB_astfly = librosa.amplitude_to_db(D_astfly, ref=np.max)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.suptitle('Log Frequency Spectrogram', fontsize=16)
# fig.delaxes(ax[1, 2])
img=librosa.display.specshow(DB_astfly, sr = sr_astfly, hop_length = hop_length, x_axis = 'time', 
                         y_axis = 'log', cmap = 'cool', ax=ax)
ax.set_title('ASTFLY', fontsize=13) 
plt.colorbar(img,ax=ax)

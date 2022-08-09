zero_astfly = librosa.zero_crossings(audio_astfly, pad=False)
zero_casvir = librosa.zero_crossings(audio_casvir, pad=False)
zero_wilfly = librosa.zero_crossings(audio_wilfly, pad=False)
zero_subfly = librosa.zero_crossings(audio_subfly, pad=False)
zero_verdin = librosa.zero_crossings(audio_verdin, pad=False)
zero_solsan = librosa.zero_crossings(audio_solsan, pad=False)
zero_birds_list = [zero_astfly, zero_casvir, zero_wilfly, zero_subfly, zero_verdin,zero_solsan]

for bird, name in zip(zero_birds_list, birds):
    print("{} change rate is {:,}".format(name, sum(bird)))

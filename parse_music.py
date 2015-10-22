from markov_lib import notes_from_wav
import pickle

# Get notes from wav
wav_list = ["minuet_in_g_major", "fur_elise.wav", "symphony_in_am.wav", "air_on_the_g_string.wav"]

notes0 = notes_from_wav(wav_list[0])
notes1 = notes_from_wav(wav_list[1])
notes2 = notes_from_wav(wav_list[2])
notes3 = notes_from_wav(wav_list[3])

with open("test_notes/notes0", 'wb') as f:
    pickle.dump(notes0, f)
with open("test_notes/notes1", 'wb') as f:
    pickle.dump(notes1, f)
with open("test_notes/notes2", 'wb') as f:
    pickle.dump(notes2, f)
with open("test_notes/notes3", 'wb') as f:
    pickle.dump(notes3, f)

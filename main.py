from markov_lib import stochastic_matrix, random_walk, song_from_notes, playMusic
from markov_lib import rate
import pickle

# import notes
with open("test_notes/notes0", 'rb') as f:
    notes0 = pickle.load(f)
with open("test_notes/notes1", 'rb') as f:
    notes1 = pickle.load(f)
with open("test_notes/notes2", 'rb') as f:
    notes2 = pickle.load(f)
with open("test_notes/notes3", 'rb') as f:
    notes3 = pickle.load(f)

# Calculate stochastic matrix
matrix = stochastic_matrix(notes0, notes1, notes2, notes3, degree=2)

generated_song_notes = random_walk(matrix, len(notes1), notes1, initial_note=[notes1[0], notes1[1]], degree=2)

# play song notes!
playMusic("generated", rate, song_from_notes(generated_song_notes, 0.2))


VECTORS = ['vx', 'vy', 'f', 'az', 'al']  # names of vectors to convert to
VECTOR_SIZE = 500
VECTOR_COUNT = len(VECTORS)
SEQUENCE_LENGTH = 10  # sequence length for stochastic learning
FIRST_LAYER = 3  # size of every encoder in first layer
SECOND_LAYER = 10  # size of second layer
KEEP_PROB = 0.6  # keep probability for droput
HIDDEN = [VECTOR_COUNT * FIRST_LAYER, SECOND_LAYER]  # structure of neural network for Network constructor

RANDOM_SETS = 50  # how many random noise patterns to add to dataset

TEST_PERCENT = 20  # how many percent of dataset becomes trainning data

NOISE_LEVEL = 0.1  # amplitude of noise to add to datasets (unused)

# savitsky-golay params
SMOOTH = True
WINDOW_SIZE = 51
ORDER = 3

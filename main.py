from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

def generate(model, tokenizer, max_length, seed_text):  # used to generate sequence from given language model
	encoded = tokenizer.texts_to_sequences([seed_text])[0]  # encode text for use in keras
	encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')  # pad to specific length we want
	yhat = model.predict_classes(encoded, verbose=0)  # predict probability for any given word
	out_word = ''
	for word, index in tokenizer.word_index.items():
		if index == yhat:
			out_word = word
			break
	return out_word

# source: Humpty Dumpty
data = 	     """Humpty Dumpty sat on a wall,\n
		Humpty Dumpty had a great fall.\n
		All the king's horses and all the king's men\n
		Couldn't put Humpty together again\n""".lower()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])  # fit a tokenizer to out source text
vocab_size = len(tokenizer.word_index) + 1

sequences = list()
for line in data.split('\n'):  # build sequences off of each line of input
	encoded = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(encoded)):
		sequence = encoded[:i+1]
		sequences.append(sequence)

max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')  # pad sequences to get length we need

sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]  # make input
y = to_categorical(y, num_classes=vocab_size)  # and output elements

model = Sequential()  # Build model!
model.add(Embedding(vocab_size, 10, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # compile
model.fit(X, y, epochs=500, verbose=1)  # fit
while True:  # and test with user input
	input = raw_input("Insert word(s) from the corpus: ")
	if input not in data or input == "":  # Error handling for bad input
		print "Error: word(s) not in corpus"
	else:
		print(generate(model, tokenizer, max_length-1, input))  # give response word!

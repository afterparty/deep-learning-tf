import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing

max_len = 200
n_words = 10000
dim_embedding = 256
EPOCHS = 20
BATCH_SIZE = 500

def load_data():
	# load data
	(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=n_words)
	# pad sequences with max_len
	x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
	x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
	return (x_train, y_train), (x_test, y_test)

def build_model():
	model = models.Sequential()
	# input: - eEmbedding Layer.
	# the model will take as input an integer matrix of size (batch, input_length)
	# the model will output dimension (input_length, dim_embedding)
	# the largest integer in the input should be no larger
	# than n_words (vocabulary size)
	model.add(layers.Embedding(n_words,
							dim_embedding, input_length=max_len))
	
	model.add(layers.Dropout(0.3))

	#takes the maximum value of either feature vector from each of the n_words features
	model.add(layers.GlobalMaxPooling1D())
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	
	return model

(x_train, y_train), (x_test, y_test) = load_data()
model = build_model()
model.summary()

model.compile(optimizer= "adam", loss= "binary_crossentropy", metrics= ["accuracy"])

score = model.fit(x_train, y_train,
				  epochs= EPOCHS,
				  batch_size= BATCH_SIZE,
				  validation_data= (x_test, y_test))

score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print("\nTest Score:", score[0])
print('Test accuracy:', score[1])
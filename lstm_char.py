from __future__ import print_function
import __builtin__
import random
import getopt
import sys
import os
import string
import json
import math
import codecs
from lib import str_manipulation as manip
from lib.logging import *
from lib.o_classes import *

# falls die Verzeichnisse noch nicht existieren sollten, werden sie erstellt
def make_dirs():
	paths = ["json/", "output/", "input/", "cfg/", "weights/", "json/ascii", "json/expanded", "json/full", "weights/ascii", "weights/expanded", "weights/full"]
	for i in paths:
		if not os.path.exists(i):
			os.makedirs(i)

# wir importieren Theano erst, wenn sichergestellt ist,
# dass die Voraussetzungen zur Erstellung/Verwendung eines Netzwerks gegeben sind			
def activate_backend():
	global Sequential, Dense, Activation, Dropout, LSTM, get_file, model_from_json, np, LossHistory
	
	log("\n[1]\nimportiere Theano...")
	
	try:
		from keras.models import Sequential
		from keras.layers.core import Dense, Activation, Dropout
		from keras.layers.recurrent import LSTM
		from keras.utils.data_utils import get_file
		from keras.models import model_from_json
		import keras.callbacks
		import numpy as np
		
		# um später den Durchschnitt errechnen zu können
		class LossHistory(keras.callbacks.Callback):
			def on_train_begin(self, logs={}):
				self.losses = []
			def on_batch_end(self, batch, logs={}):
				self.losses.append(logs.get('loss'))
	except ImportError:
		log("Module nicht gefunden! Wurden Theano und Keras installiert?")
			
# Vektorisierung
def vectorize():
	global chars, char_indices, indices_char, X, y, dimensionality
	
	log("\n[2]\nvektorisiere...")
	
	chars, dimensionality = manip.get_charset_and_dimensionality(text.charset)

	char_indices = dict((c, i) for i, c in enumerate(chars)) # Indizierung
	indices_char = dict((i, c) for i, c in enumerate(chars))
	
	# der Text wird alle drei Zeichen in 40 Zeichen lange "Sätze" zerlegt
	step = 3
	sentences = []
	next_chars = []
	for i in range(0, text.limit - scope, step):
		sentences.append(text.content[i: i + scope])
		next_chars.append(text.content[i + scope])
	assert int(np.ceil(float(text.limit - scope)/3.0)) == len(sentences), "Zeichenlimit und Textlänge stimmen nicht überein!"
	
	if verbose:
		log("Sequenzen:", len(sentences))
		log("Zeichenlimit:", text.limit)
		log("Ausschnitt:\n", text.print_snippet(500))
		log("Zeichenvorrat:", text.charset)
		log("Dimensionalität:", dimensionality)
	
	# die Vektoren werden erstellt
	X = np.zeros((len(sentences), scope, dimensionality), dtype=np.bool)
	y = np.zeros((len(sentences), dimensionality), dtype=np.bool)
	for i, sentence in enumerate(sentences):
		for t, char in enumerate(sentence):
			X[i, t, char_indices[char]] = 1
		y[i, char_indices[next_chars[i]]] = 1
	
# Modellerstellung		
def build_model():
	global model
	
	log("\n[3]\nerstelle/lade Modell...")
	
	# ein neues Modell wird erstellt oder ein altes geladen
	if fresh:
		model = Sequential()
		model.add(LSTM(512, return_sequences=True, input_shape=(scope, dimensionality)))
		model.add(Dropout(0.2))
		model.add(LSTM(512, return_sequences=False))
		model.add(Dropout(0.2))
		model.add(Dense(dimensionality))
		model.add(Activation('softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
		
		# das Modell wird im JSON-Format gespeichert
		with open("json/" + text.charset + "/" + text.name + ".json", "w") as file:
			file.write(model.to_json())
	else:
		# falls die Gewichtung eines anderen Textes geladen werden soll
		if other_weights:
			log("Welche Gewichtung soll geladen werden?")
			selection = raw_input("\a> ")
			if selection.endswith(".hdf5"):
				selection = selection[0:-5]
			
			path = "weights/" + text.charset + "/" + str(text.limit) + "/" + selection + ".hdf5"
			if not os.path.exists(path):
				log("Die angegebene Gewichtung existiert nicht!")
				sys.exit(2)
			else:
				text.set_weights(selection)
				with open("json/" + text.charset + "/" + text.weights + ".json") as file:
					model = model_from_json(file.read())
				model.load_weights(path)
				model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
				log("Soll das Modell gespeichert werden?(J/n)")
				answer = raw_input("\a> ")
				if answer in "J":
					# das Modell wird im JSON-Format gespeichert
					with open("json/" + text.charset + "/" + text.weights + ".json", "w") as file:
						file.write(model.to_json())
						log("...Modell gespeichert")
		else:
			with open("json/" + text.charset + "/" + text.name + ".json") as file:
				model = model_from_json(file.read())
			model.load_weights("weights/" + text.charset + "/" + str(text.limit) + "/" + text.name + ".hdf5")
			model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
		
# Training
def train_network():
	global history
	
	history = LossHistory()
	
	log("\n[4]\nTraining läuft...", "\nDurchlauf:", str(text.iteration))
	model.fit(
		X,
		y,
		batch_size=128,
		nb_epoch=text.training,
		verbose=1,
		callbacks=[history]
	)

	text.set_iteration(text.iteration + 1)
	
# Textgenerierung
def generate_text():

	if generate_only:
		log("\n[4]\nGenerierung läuft...\n")
	else:
		log("\n[5]\nGenerierung läuft...\n")
		
	start_index = random.randint(0, len(text.content) - scope - 1)
	sentence = text.content[start_index: start_index + scope]
	generated = ""
	generated += sentence
	errors = 0
	
	# hier wird live generiert
	for i in range(output_length):
		x = np.zeros((1, scope, dimensionality))
		for t, char in enumerate(sentence):
			x[0, t, char_indices[char]] = 1.
		preds = model.predict(x, verbose=0)[0]
		next_index = sample(preds, diversity)
		next_char = indices_char[next_index]
		generated += next_char
		sentence = sentence[1:] + next_char
		try:
			sys.stdout.write(next_char)
			sys.stdout.flush()
		except UnicodeEncodeError as err:
			errors += 1
			
	log("\nFehleranzahl:", errors)
	
	save_text(generated)

# Textgenerierung mit Seed
def generate_with_arbitrary_seed():

	if generate_only:
		log("\n[4]\nWelcher Satz soll der Generierung zugrunde liegen?")
		seed = raw_input("\a> ")
	else:
		log("\n[5]\nWelcher Satz soll der Generierung zugrunde liegen?")
		seed = raw_input("\a> ")
	
	# wir gewährleisten, dass der Seed mit dem Netzwerk kompatibel ist
	seed_dec = seed.decode(sys.stdin.encoding)
	if text.charset in "ascii":
		sentence = manip.convert_to_ascii(seed_dec)
	elif text.charset in "expanded":
		sentence = manip.convert_to_expanded(seed_dec)
	elif text.charset in "full":
		sentence = seed_dec
	if len(sentence) > scope:
		sentence = sentence[-scope:]
		
	generated = ""
	vector_difference = scope - len(sentence)
	errors = 0
	
	log("\n(das Netzwerk antwortet)\n")
	
	for i in range(output_length):
		x = np.zeros((1, scope, dimensionality))
		for t, char in enumerate(sentence):
			x[0, t + vector_difference, char_indices[char]] = 1.
		
		preds = model.predict(x, verbose=0)[0]
		next_index = sample(preds, diversity)
		next_char = indices_char[next_index]
		
		generated += next_char
		sentence = sentence[1:] + next_char
		try:
			sys.stdout.write(next_char)
			sys.stdout.flush()
		except UnicodeEncodeError as err:
			errors += 1
			
	log("\nFehleranzahl:", errors)
	
	save_text(generated)	
			
# Speichern der Ausgabe
def save_text(txt):

	if generate_only:
		log("\n\n[5]\nspeichere generierten Text...")
	else:
		log("\n\n[6]\nspeichere generierten Text...")
		
	directory = 'output/' + text.name + '/'
	if not os.path.exists(directory):
		os.makedirs(directory)
	random_seed = manip.generate_seed()
	output_name = text.charset + "_" + random_seed + '.txt'
	with open(directory + output_name, 'w') as file:
		if text.charset in "full":
			txt = txt.encode("utf-8", "ignore")
		file.write(txt)
	
	log("Die Ausgabe findest du hier:", directory + output_name)
		
def sample(a, temperature=1.0):
	a = np.log(a) / temperature
	a = np.exp(a) / np.sum(np.exp(a))
	return np.argmax(np.random.multinomial(1, a, 1))
	
# das Programm wird konfiguriert	
def check_arguments(opts):
	global input, fresh, resume, no_of_epochs, other_weights, character_limit, generate_only, output_length, diversity, arbitrary_seed, mutual_exclusives, dyn_range, verbose, ascii, expanded
	
	input = None; fresh = False; resume = False # Training allgemein
	no_of_epochs = 10; other_weights = False; character_limit = 100000; dyn_range = False # Training im Speziellen
	generate_only = False; output_length = 500; diversity = 0.2; arbitrary_seed = False # Textgenerierung
	mutual_exclusives = 0; verbose = False; ascii = False; expanded = False
	
	for o, a in opts:
		if o in ("-h", "--help"):
			usage()
			sys.exit(2)
		elif o in ("-i", "--input"):
			input = str(a)
		elif o in ("-n", "--new"):
			fresh = True
			mutual_exclusives += 1
		elif o in ("-r", "--resume"):
			resume = True
			mutual_exclusives += 1
		elif o in ("-g", "--generate"):
			generate_only = True
			output_length = int(a)
			mutual_exclusives += 1
		elif o in ("-d", "--diversity"):
			diversity = float(a)
			if diversity > 2.0:
				diversity = 2.0
		elif o in ("-e", "--epochs"):
			no_of_epochs = int(a)
		elif o in ("-o", "--other"):
			other_weights = True
		elif o in ("-l", "--limit"):
			if str(a) in "max":
				character_limit = "max"
			else:
				character_limit = int(a)
		elif o in ("-s", "--seed"):
			arbitrary_seed = True
		elif o in ("-y", "--dynamic"):
			dyn_range = True
		elif o in ("-v", "--verbose"):
			verbose = True
		elif o in ("-a", "--ascii"):
			ascii = True
		elif o in ("-p", "--expanded"):
			expanded = True
		else:
			assert False, "Die Option existiert nicht!"

# Programm			
def main():
	global text, character_limit, scope
	scope = 40 # ich nenne das mal "scope", notfalls kann man den Wert dann hier verändern
	
	# die Argumente der Befehlszeile werden eingelesen
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hi:nrg:d:e:ol:syvap", ["help", "input=", "new", "resume", "generate=", "diversity=", "epochs=", "other", "limit=", "seed", "dynamic", "verbose", "ascii", "expanded"])
	except getopt.GetoptError as err:
		print(str(err))
		usage()
		sys.exit(2)
	check_arguments(opts)
	
	# wir prüfen, ob die gewählten Optionen stimmig sind
	assert type(input) is str, "Ausgangstext muss angegeben werden!"
	assert mutual_exclusives <= 1, "Die gewählten Verfahren sind nicht kombinierbar!"
	assert ascii and not expanded or expanded and not ascii or not ascii and not expanded, "Mehrere Zeichensätze ausgewählt!"
	
	make_dirs() # falls die Verzeichnisse noch nicht erstellt wurden
	
	dec_print("=LSTM_CHAR.PY")	
	log("öffne und konvertiere", input + "...")
	
	# Textdatei wird eingelesen und konvertiert
	try:
		path = "input/" + input
		if not os.path.isfile(path):
			path = input
		with open(path) as file:
			if ascii:
				input_c = manip.convert_to_ascii(file.read())
			elif expanded:
				input_c = manip.convert_to_expanded(file.read())
			else:
				input_c = manip.convert_to_full(file.read())
	except EnvironmentError:
		print("Datei {0} nicht gefunden!".format(input))
		sys.exit(2)
	
	if type(character_limit) is str and character_limit in "max":
		character_limit = len(input_c)
	assert character_limit <= len(input_c), "Zeichenlimit zu hoch!"
	
	if dyn_range and character_limit < len(input_c):
		log("Wir probieren die dynamic range aus...")
		input_c = manip.get_dynamic_range(input_c, character_limit)
	
	# neuer Anfang? Wiederaufnahme? andere Konfiguration?
	if resume or generate_only:
		text = TextObject(manip.truncate(input), input_c, resuming=True)	
	else:
		if ascii:
			text = TextObject(manip.truncate(input), input_c, "ascii", limit=character_limit, training=no_of_epochs)
		elif expanded:
			text = TextObject(manip.truncate(input), input_c, "expanded", limit=character_limit, training=no_of_epochs)
		else:
			text = TextObject(manip.truncate(input), input_c, "full", limit=character_limit, training=no_of_epochs)
	if generate_only:
		log("Es wird, ausgehend von der gespeicherten Gewichtung (Zeichenlimit {0}), ein {1} Zeichen langer Text mit diversity-Grad {2} generiert.".format(character_limit, output_length, diversity))
	else:
		if fresh:
			log("Sicher, dass von 0 an trainiert werden soll?(J/n)")
			answer = raw_input("\a> ")

		log("Das Netzwerk wird {0} Epoche(n) lang trainiert.".format(text.training))
		
	if resume:
		log("\nEs wird mit folgender Konfiguration weitertrainiert:")
		log("Zeichenlimit:", text.limit)
		log("Zeichensatz:", text.charset)
		log("Epochenanzahl:", text.training)
		log("Durchlauf:", text.iteration)
		log("Verlust:", text.loss)
		log("\nIst das so richtig?(J/n)")
		answer = raw_input("\a> ")
		if answer in "J":
			log("Dann fahren wir fort.")
		else:
			log("Vorgang wird abgebrochen.")
			sys.exit(2)
			
	activate_backend() # Theano wird importiert
	vectorize() # die Vektoren werden erstellt
	build_model() # das Modell wird erstellt oder zusammen mit der gespeicherten Matrix geladen
	
	# trainieren oder generieren?
	if not generate_only:
		train_network() # das Netzwerk wird trainiert
		
		path = "weights/" + text.charset + "/" + str(text.limit) + "/"
		if not os.path.exists(path):
			os.makedirs(path)
		if not other_weights:
			model.save_weights(path + text.name + '.hdf5', overwrite=True)
		else:
			log("\nSoll die Gewichtung {0} überspeichert werden?(J/n)".format(path + text.weights + ".hdf5"))
			answer = raw_input("\a> ")
			if answer in "J":
				model.save_weights(path + text.weights + '.hdf5', overwrite=True)
			elif answer in "n":
				log("\nSoll die Gewichtung stattdessen als {0} gespeichert werden?(J/n)".format(path + text.name + ".hdf5"))
				answer_c = raw_input("\a> ")
				if answer_c in "J":
					model.save_weights(path + text.name + ".hdf5", overwrite=True)
			else:
				seed = manip.generate_seed()
				backup = path + text.name + "_" + seed + ".hdf5"
				model.save_weights(backup)
				log("Gewichtung wurde unter {0} gespeichert.".format(backup))
		
		# der durchschnittliche Verlust wird errechnet
		losses_sum = 0.0
		losses_r = 0.0
		for i in history.losses:
			losses_sum += float(i)
			losses_r =  round(losses_sum / len(history.losses), 10)
		log('Durchschnitt:', losses_r)
	
		text.save_config(losses_r) # die Konfiguration wird gespeichert
	
	# Textgenerierung
	if arbitrary_seed:
		generate_with_arbitrary_seed()
	else:
		generate_text()
		
	dec_print("=LSTM_CHAR.PY SAGT DANKE!")
	
# Hilfe
def usage():
	try:
		with open("readme.txt") as file:
			log(file.read())
	except IOError as err:
		print(err)

if __name__ == "__main__":
	main()
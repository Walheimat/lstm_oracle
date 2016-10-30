from __future__ import print_function
import random
import time
import string

printable_exp = string.printable + "äöüÄÖÜß" # unser erweiterter ASCII-Zeichensatz
printable_exp_quote = "»«›‹„“‚‘"
verbose = False

def alter_mood(f):
	global mood
	
	mood_swing = random.randint(-1, 2) # Stimmungsschwankungen
	if verbose:
		print("mood_swing:", mood_swing)
	
	if f > tipping_point:
		degree = int((f - tipping_point) / 0.2) # in "negative" Richtung ist ein Anstieg von 1 bis 5 möglich
		mood = min(mood + degree + mood_swing, 20)
		assert mood <= 20, "mood liegt bei:" + str(mood)
	elif f <= tipping_point:
		degree = int((tipping_point - f) / 0.4) * -1 # in "positive" Richtung ist eine Verringerung von 1 oder 2 möglich
		mood = max(mood + degree + mood_swing, 1)
		assert mood >= 1, "mood liegt bei:" + str(mood)
		
	return moods[mood]

def get_proximity_as_diversity_and_mood(s):

	word_list = remove_punctuation(s).lower().split()	
	word_count = len(word_list)
	weight_count = 0
	
	for w in word_list:
		if w in weighted_words:
			to_add = weighted_words[w]
			if to_add > 1000:
				weight_count += 10
			elif to_add > 100:
				weight_count += 5
			elif to_add > 10:
				weight_count += 2
			else:
				weight_count += to_add
		else:
			weight_count += 1
	
	proximity = (word_count / float(weight_count)) * 2
	assert proximity >= 0.2, "Rechenfehler!"
	mood_str = alter_mood(proximity)
	if verbose:
		print("proximity:", proximity)
		print("mood:", mood)
	proximity = min((proximity * (mood/10.0)) + 0.18, 2.0)
	assert proximity <= 2.0, "Rechenfehler!"
	
	return proximity, mood_str

def create_word_table(txt, verbosity):
	global weighted_words, mood, tipping_point, moods, verbose
	
	verbose = verbosity
	mood = 10 # Anfangswert, Maximum ist 2.0, Minimum 0.1
	tipping_point = 1.0
	moods = {
		1:"verliebt in dich", 2:"freundschaftlich", 3:"freundschaftlich", 4:"freundlich", 5:"freundlich", 6:"freundlich",
		7:"wohlgesonnen", 8:"wohlgesonnen",
		9:"wohlgesonnen", 10:"neutral", 11:"ungeduldig",
		12:"ungeduldig", 13:"ungeduldig", 14:"schroff",
		15:"schroff", 16:"verärgert", 17:"verärgert",
		18:"wütend", 19:"wütend",
		20:"wahnsinnig vor Wut"}

	weighted_words = {}
	w_array = remove_punctuation(txt).split()
	
	for word in w_array:
		if word in weighted_words:
			weighted_words[word] += 1
		else:
			weighted_words[word] = 1

# für das wortbasierte Netzwerk
def convert_to_wordlist(txt, bom_removal=True):
	punctuation = get_punctuation()
	converted = ""
	if bom_removal:
		txt = remove_bom(txt)
	txt = txt.replace("\n", " ")
	
	# white space vor Satzzeichen einfügen
	for c in txt:
		if c in punctuation:
			converted += " "
		converted += c
	
	converted_split = converted.split(" ")
	return list(converted_split)

def concatenate_wordlist(lst):
	punctuation = get_punctuation()
	converted = ""
	
	for w in lst:
		if w not in punctuation:
			converted += " "
		converted += w
	
	return converted
	
def get_punctuation():
	punctuation = ""
	
	with open("docs/punctuation") as file:
		for line in file:
			if not line.startswith("#"):
				punctuation += line.strip()
	
	assert len(punctuation) is 9, "Interpunktion zu kurz/lang! Soll: 9, Haben: " + str(len(punctuation))
	
	return punctuation
	
def get_full_charset():
	global number_of_unique_chars
	
	full_charset = ""
	
	# als erstes werden die white-space-Zeichen hinzugefügt
	full_charset += "\n ".decode("utf-8")
	
	# der externe Zeichensatz wird geladen
	with open("docs/symbols") as file:
		for line in file:
			if not line.startswith("#"):
				full_charset += line.strip().decode("utf-8")
				
	assert len(full_charset) == 160, "Zeichenvorrat zu kurz/lang!"
	assert len(list(full_charset)) == 160, "Länge der Liste nicht korrekt!"
	
	return list(full_charset)
	
def convert_to_full(txt):
	full_charset = get_full_charset()
	number_of_unique_chars = len(full_charset)
	text = remove_bom(txt)
	decoded = list(text.decode("utf-8"))
	out = []
	for c in decoded:
		if c in full_charset:
			out.append(c)
	new_out = "".join(out)

	return new_out, number_of_unique_chars
	
def generate_seed():
	sequence = list(string.hexdigits)
	random.shuffle(sequence)
	current_time = time.strftime("%d-%m-%Y")
	return "".join(sequence)[0:4] + "_" + current_time
	
def get_charset_and_dimensionality(c):
	if c in "expanded":
		return printable_exp, len(printable_exp)
	elif c in "ascii":
		return string.printable, len(string.printable)
	elif c in "full":
		return get_full_charset(), 200

# entfernt die byte order mark
def remove_bom(txt):
	i = txt.decode("utf-8-sig")
	u = i.encode("utf-8")
	return u
	
def remove_punctuation(txt):
	punc = get_punctuation()
	txt_conv = ""
	for c in txt:
		if c not in punc:
			txt_conv += c
	return  txt_conv

# konvertiert den Text	
def convert_to_expanded(txt):
	u = ""
	lchar = ''
	
	for c in txt:
		if c in printable_exp:
			u += c
		elif c in printable_exp_quote and lchar in printable_exp_quote:
			u += '"'
		lchar = c
	
	return u

def convert_to_ascii(txt):
	o = txt.decode("utf-8")
	u = o.encode("ascii", "ignore")
	return u

# Name ohne Dateiendung
def truncate(str):
	i = str.partition(".")
	return i[0]

# damit auch bei gesetztem Zeichenlimit zumindest theoretisch der Gesamttext berücksichtigt wird
def get_dynamic_range(str, lmt):
	length = len(str)
	buffer = length - lmt
	starting_point = random.randint(0, buffer)
	
	return str[starting_point: starting_point + lmt]
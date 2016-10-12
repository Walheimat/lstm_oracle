from __future__ import print_function

import random
import time
import string

printable_exp = string.printable + "äöüÄÖÜß" # unser erweiterter ASCII-Zeichensatz
printable_exp_quote = "»«›‹„“‚‘"

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
	
	with open("punctuation") as file:
		for line in file:
			if not line.startswith("#"):
				punctuation += line.strip()
	
	assert len(punctuation) is 9, "Interpunktion zu kurz/lang! Soll: 9, Haben: " + str(len(punctuation))
	
	return punctuation
	
def get_full_charset():
	full_charset = ""
	
	# als erstes werden die white-space-Zeichen hinzugefügt
	full_charset += "\n ".decode("utf-8")
	
	# der externe Zeichensatz wird geladen
	with open("symbols") as file:
		for line in file:
			if not line.startswith("#"):
				full_charset += line.strip().decode("utf-8")
				
	assert len(full_charset) == 160, "Zeichenvorrat zu kurz/lang!"
	assert len(list(full_charset)) == 160, "Länge der Liste nicht korrekt!"
	
	return list(full_charset)
	
def convert_to_full(txt):
	full_charset = get_full_charset()	
	text = remove_bom(txt)
	decoded = list(text.decode("utf-8"))
	out = []
	for c in decoded:
		if c in full_charset:
			out.append(c)
	new_out = "".join(out)
	
	return new_out
	
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
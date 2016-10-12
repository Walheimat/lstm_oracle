from __future__ import print_function
import sys

# gewährleistet, dass in der Windows-Konsole Zeichen richtig angezeigt werden		
def log(*args, **kwargs):
	if sys.platform == "win32":
		elements = list(args)
		str_elements = []
		for e in elements:
			if not isinstance(e, unicode):
				str_elements.append(str(e))
			else:
				k = e.encode("utf-8")
				str_elements.append(str(k))
		s = " ".join(str_elements)
		print(s.decode("utf-8"))
	else:
		print(*args, **kwargs)

# nur zum Ausprobieren		
def symbol_decoration(func):
	def inner(string):
		decsym = string[:1]
		to_print = string[1:]
		print()
		print(len(to_print) * decsym)
		func(to_print)
		print(len(to_print) * decsym)
		print()
	return inner

@symbol_decoration
def dec_print(msg):
	log(msg)
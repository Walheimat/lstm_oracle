import json
import os
import random

# hauptsächlicher Zweck ist, die Konfiguration zu speichern/laden
# und die Zahl der Durchläufe pro Zeichenlimit festzuhalten
class TextObject:

	def __init__(self, name, content, charset="undefined", limit=10000, training=10, resuming=False):
		self.name = name
		self.content = content
		
		if not resuming:
			self.training = training
			self.limit = limit
			self.weights = self.name
			self.charset = charset
			self.iterations = self._get_iterations()
			if str(self.limit) in self.iterations:
				self.iteration = self.iterations[str(self.limit)]
			else:
				self.iteration = 1
		else:
			self._load_config()
		
	def set_weights(self, str):
		self.weights = str
		
	def set_iteration(self, int):
		self.iterations[str(self.limit)] = int
		
	def _get_iterations(self):
		try:
			if self.charset is "undefined":
				path = "cfg/word/" + self.name + "/config.json"
			else:
				path = "cfg/" + self.name + "/config.json"
			if os.path.isfile(path):
				with open(path) as file:
					data = json.load(file)
					iterations = data["iterations"]
					return iterations
			else:
				return {}
		except IOError as err:
			print(err)
		
	def save_config(self):
		try:
			if self.charset is "undefined":
				path ="cfg/word/" + self.name + "/"
			else:
				path = "cfg/" + self.name + "/"
			if not os.path.exists(path):
				os.makedirs(path)	
			data ={"weights" : self.weights, "limit" : self.limit, "training" : self.training, "iterations" : self.iterations, "charset" : self.charset}
			with open(path + "config.json", "w") as file:
				json.dump(data, file)
		except IOError as err:
			print(err)
			
	def _load_config(self):
		try:
			path = "cfg/" + self.name + "/config.json"
			if os.path.isfile(path):
				with open(path) as file:    
					data = json.load(file)
					self.training = data["training"]
					self.limit = data["limit"]
					self.weights = data["weights"]
					self.iterations = data["iterations"]
					self.charset = data["charset"]
					self.iteration = self.iterations[str(self.limit)]
			else:
				log("Datei existiert nicht!")
		except IOError as err:
			print(err)
			
	def print_snippet(self, arg):
		starting_point = random.randint(0, len(self.content) - arg)
		return self.content[starting_point:starting_point + arg]
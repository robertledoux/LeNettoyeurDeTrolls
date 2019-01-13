import json
import math
from pprint import pprint
import random
import numpy as np

def nombreDeMots(txt):
	return len(txt.split(" "))

def longueurDuTexte(txt):
	return len(txt)

def pontuactionAbusive(txt):
	interrogation_complexe = txt.count("??")
	exclamation_complexe = txt.count("!!")
	return interrogation_complexe + exclamation_complexe + 1

def majusculeExcessive(txt):
	mots = txt.split(" ")
	compteur = 0
	for m in mots:
		if m.istitle():
			compteur = compteur + 1
	return compteur/len(mots)

def respectSujet(txt):
	sujet = "L'Informateur de La Révélation des Pyramides est notre invité. Nous le questionnons sur sa méthode de travail : comment établit-il les faits sur lesquels il se fonde ? Comment teste-t-il ses interprétations ? Sa démarche permet-elle de valider les découvertes qu'il prétend avoir faites. Le dialogue est difficile, et Jacques Grimault n'accorde que de très rares réponses à nos questions... (Du coup se pose la question : pourquoi venir dans une émission comme la nôtre si ce n'est pas pour répondre aux questions et aux critiques ?).".split(" ")
	temp_sujet = ""
	for s in sujet:
		if len(s) > 3:
			temp_sujet = temp_sujet + " " + s
	sujet = temp_sujet.lower().split(" ")
	txt = txt.lower().split(" ")
	return len([value for value in sujet if value in txt])

def insultant(txt):
	insultes = ["pd", "enculé", "salope", "connard", "pédé", "salaud", "pute"]
	txt = txt.lower().split(" ")
	if len([value for value in insultes if value in txt]) == 0:
		return 1
	else:
		return -1000000

def remerciement(txt):
	remerciement = ["merci", "bonne continuation"]
	txt = txt.lower().split(" ")
	if len([value for value in remerciement if value in txt]) == 0:
		return 1
	else:
		return -math.log(len([value for value in remerciement if value in txt])/len(txt))


with open("comments.json", encoding="utf8") as f:
    data = json.load(f)






index = 0
X = []
Y = []


for d in data:

	text = d["commentText"]
	if d["numberOfReplies"] == 0:
		d["numberOfReplies"] = 1.5

	ans = int(input(d["commentText"] + "\n"))
	if ans != 9:
		points = [nombreDeMots(text), d["likes"], longueurDuTexte(text), pontuactionAbusive(text), majusculeExcessive(text), respectSujet(text), d["numberOfReplies"], insultant(text), remerciement(text)]
		X.append(points)		
		Y.append([ans])
	else:
		break






# X = (hours studying, hours sleeping), y = score on test, xPredicted = 4 hours studying & 8 hours sleeping (input data for prediction)
X = np.array(X, dtype=float)
y = np.array(Y, dtype=float)
xPredicted = np.array(([4,5,8,4,8,4,8,4,8]), dtype=float)

# scale units
X = X/np.amax(X, axis=0) # maximum of X array
xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)
y = y/100 # max test score is 100

class Neural_Network(object):
  def __init__(self):
  #parameters
    self.inputSize = 9
    self.outputSize = 1
    self.hiddenSize = 9

  #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propagate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

  def predict(self):
    print("Predicted data based on trained weights: ")
    print("Input (scaled): \n" + str(xPredicted))
    print("Output: \n" + str(self.forward(xPredicted)))

NN = Neural_Network()
for i in range(0,1000): # trains the NN 1,000 times
  print("# " + str(i) + "\n")
  print("Input (scaled): \n" + str(X))
  print("Actual Output: \n" + str(y))
  print("Predicted Output: \n" + str(NN.forward(X)))
  print("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
  print("\n")
  NN.train(X, y)

NN.saveWeights()
NN.predict()
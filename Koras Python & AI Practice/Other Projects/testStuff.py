import random
from Challenges import squareNum
import numpy as np

numListidk = [10,9,8,7]
print(squareNum(numListidk))
print("Hello World")
idk = 5
idk2 = 6.0 #double, long etc
bored = True
name = "Stewart"
firstInitial = 's'

if bored:
    print(name + " is bored...")
elif not bored: #else
    print(":)")

fruitList = ["apple", "orange", "pear", "banana", "dragonfruit", "kiwi"]
for fruit in fruitList:
    print(fruit + " is a fruit")

numFruit = 3
print("then...")
for fruit in range(0, numFruit):
    print(fruitList[fruit] + " is a fruit")

x = 0
while x < 3:
    print(random.randint(0,10))
    x += 1

def Stews9Multiplication(inputNum):
    tensPlace = inputNum - 1
    onesPlace = 10 - inputNum
    if abs(inputNum) <= 10:
        print(str(tensPlace) + str(onesPlace))
    else:
        print("number is invalid")

def random_number(rangeNum):
    return random.randint(0, rangeNum)

def printList(newList): #keep
    for x in newList:
        print(x)

def sortList(newList):
    newList.sort
    printList(newList)

class MyPet:
    name = ""
    age = 0
    type = ""
    def __init__(self, newName, newAge, newType):
        name = newName
        age = newAge
        type = newType
    def speak(self):
        print(name)
#Maxine = MyPet("Maxine",4,"cat")
class MyCat(MyPet):
    def speak(self): #override for specific subclass
        print("meow")
        print("meow")

"""class Card:
    suit = ""
    type = ""
    def __init__(self, newSuit, newType):
        suit = newSuit
        type = newType
    def cardName(self):
        return type + " of " + suit
suitList = ["heart", "diamond", "spade", "clubs"]
typeList = []
for x in range(2, 11):
    typeList.append(str(x))
typeList.append("jack")
typeList.append("queen")
typeList.append("king")
typeList.append("ace")

cardList = []
for suit in suitList:
    for type in typeList:
        card = Card(suit, type)
        cardList.append(card)
    #     cardList.append(Card(suit, type))

for card in cardList:
    print(card.cardName())
    """

fancyPantsList = [[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]]] #4x3x2 array
counter = 0
for x in fancyPantsList:
    for y in x:
        for z in y:
            counter += 1
print(counter)
print(np.shape(fancyPantsList))
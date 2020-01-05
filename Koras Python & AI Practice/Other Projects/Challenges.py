
def testConditionial(x):
    if x < 5:
        print("yes")
    else:
        print("no")
#testConditional(10)

def squareNum(numList):
    for num in numList:
        print(num * num)
#y = [1,2,3,4,5]
#squareNum(y)
def idk(x):
    if (x*3) > 90:
        return True
    else:
        return False
def otheridkLOL(x):
    if (x*3)%2 == 1:
        return True
    else: return False
#print(idk(5))

def defVowelizer(word):
    isVow = False
    retWord = ""
    vows = ["a","e","i","o","u"]
    for letter in range(0, len(word)):
        for check in vows:
            if word[letter] == check:
                isVow = True
        if isVow == False:
            retWord += word[letter]
        isVow = False
    return retWord
def reverser(word):
    retWord = ""
    for letter in range(0, len(word)):
        retWord += word[len(word)-letter-1]
    return retWord
# word = "Hello this is a big long test string"
# print(defVowelizer(word))
# print(reverser(word))

def recFactorial(num):
    if num == 1:
        return 1
    else:
        return num * recFactorial(num-1)
# newNum = 6
# print(recFactorial(newNum))

def isPal(word):
    if word == reverser(word):
        return True
    else:
        return False
# print(isPal("toyot"))

def countClump(word):
    counter = 0
    x = 0
    while(x < len(word)-1):
        if(word[x] == word[x+1]):
            counter += 1
            if not (x == len(word)-2):
                while (word[x] == word[x + 1]):
                    x += 1
        x += 1
    return counter
# testerClump = "123445667"
# testerClump2 = "112113333633411174556744544"
# print(countClump(testerClump2))


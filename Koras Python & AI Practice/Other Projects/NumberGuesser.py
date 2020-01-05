import random

myNumber = random.randint(0,101)
user_input = ""
def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

counter = 0
while not user_input == myNumber:
    print("Pick a number between 0 and 100...")
    # user_input = random.randint(0, 101) # if you want to watch a computer play itself
    # print(user_input)
    user_input = input()
    counter += 1
    if RepresentsInt(user_input) == False:
        print("input is invalid")

    else:
        user_input = int(user_input)
        if (user_input < myNumber):
            print("Guess again...")
            print("my number is higher")
        elif (user_input > myNumber):
            print("Guess again...")
            print("My number is lower")
        elif (user_input == myNumber):
            print("You Win!")
            print("it only took ya, " + str(counter) + " tries ya doof...")

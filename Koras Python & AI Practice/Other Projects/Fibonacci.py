from NumberGuesser import RepresentsInt
while (1<2):
    num = ""
    while RepresentsInt(num) == False:
        print("Which number in the fib sequence do you want?")
        num = input()
        if RepresentsInt(num) == False:
            print("input is invalid")
    newNum = int(num)
    a = 1
    b = 1
    while newNum >= 0:
        total = a + b
        a = b
        b = total
        newNum -= 1
    print(a)
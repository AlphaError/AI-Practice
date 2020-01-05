#By: Stewart H.

board = [[0,0,0],[0,0,0],[0,0,0]]
testboard1 = [[1,1,1],[0,0,0],[0,0,0]] # across
testboard2 = [[1,0,0],[1,0,0],[1,0,0]] # down
wowTestBoard1 = [[1,1,1],[1,1,1],[1,1,1]] # all
wowTestBoard2 = [[1,0,0],[0,1,0],[0,0,1]] # diagonal 1
wowTestBoard3 = [[1,0,0],[0,1,0],[0,0,1]] # diagonal 2

#3x3 board
#1 is player and 2 is player 2
def checkWon(board, player):
    counter = 0
    for x in range(0, 3):  # checks across
        for y in range(0, 3):
            if board[x][y] == player:
                counter += 1
        if counter >= 3:
            return True
        counter = 0
    for x in range(0, 3):  # checks down
        for y in range(0, 3):
            if board[y][x] == player:
                counter += 1
        if counter >= 3:
            return True
        counter = 0
    for x in range(0, 3):  # checks diagonal
        if board[x][x] == player:
            counter += 1
    if counter >= 3:
        return True
    counter = 0
    for x in range(0, 3):  # checks other diagonal
        if board[2 - x][2 - x] == player:
            counter += 1
    if counter >= 3:
        return True
    return False

spacing = "     "
dashes = "-------------"
gameWon = False
player = 1
player_input = "1,1"

print("")
print("Welcome to tic tac toe (3x3). You know the rules. Just input coordinates with a divider in beteen them in y,x format!") # intro
print("                                   ~~~~~ Good luck to all players! ~~~~~")
print("")
print("~(^.^)~ ✿ ~(^.^)~ ✿ ~(^.^)~ ✿ ~(^.^)~ ✿ ~(^.^)~ ✿ ~(^.^)~ ✿ ~(^.^)~ ✿ ~(^.^)~ ✿ ~(^.^)~ ✿ ~(^.^)~ ✿ ~(^.^)~ ✿ ~(^.^)~ ")
print("")

while not gameWon:
    print("Player " + str(player) + ", make your move in terms of coordinates: (example: 1,1 is the first square)")
    player_input = input()
    if board[int(player_input[0])-1][int(player_input[2])-1] != 0:
        print("The chosen space has already been played on... Try choosing another one")
    else:
        board[int(player_input[0])-1][int(player_input[2])-1] = player #updates board

        print(spacing + dashes)
        for x in range(0, 3): # prints board
            print0 = ""
            print1 = ""
            print2 = ""
            if board[x][0] == 0:
                print0 = " "
            elif board[x][0] == 1:
                print0 = "X"
            else:
                print0 = "O"
            if board[x][1] == 0:
                print1 = " "
            elif board[x][1] == 1:
                print1 = "X"
            else:
                print1 = "O"
            if board[x][2] == 0:
                print2 = " "
            elif board[x][2] == 1:
                print2 = "X"
            else:
                print2 = "O"
            print(spacing + "| " + print0 + " | " + print1 + " | " + print2 + " |")
            print(spacing + dashes) #spacing stuff
        print("")

        if checkWon(board, player) == True:
            break
        else:
            if player == 1: #reset
                player = 2
            else:
                player = 1
print("✿ Congratulations, player " + str(player) + " wins!!!  ~(^.^)~  ~(^.^)~  ~(^.^)~ ")

'''     
         \  /
          \/
          /\
         /  \
         __
        /  \
       |    |
       |    |
        \__/
'''

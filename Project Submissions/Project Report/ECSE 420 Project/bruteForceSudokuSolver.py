from multiprocessing.pool import Pool
from datetime import datetime
import sys
import math

#cd Documents\FALL 2020\ECSE 420\Project
#python testing.py HardInput2.txt 32

"""
Goal: Load the board from the file and return a list of lists 
Arguments: A file
Return: a 9x9 Sudoku Board
"""
def loadBoard(file):
    f= open(file,'r')
    inputString= f.read()
    print("The Loaded Board looks like this:")
    print(inputString)
    board = [[None] * 9 for i in range(9)]
    for y, row in enumerate(inputString.split('\n')):
        for x, cell in enumerate([a for a in list(row) if a not in [',', ' ']]):
            board[y][x] = int(cell) if cell.lower() not in ['x', '0'] else None
    return board

"""
Goal: Print a given board
Arguments: A 9x9 Sudoku Board
Return: Print the given Board to console 
"""
def printBoard(board):
    for y in range(9):
        print("".join(map(str, board[y])))

"""
Goal: Takes in a sudoku board solve it using pultiple processors
Arguments: A 9x9 Sudoku Board, and number of processors to be used
Return: The solution to the initial given board
"""
def algorithm(board, num_processors):
    board_copy = [[None] * 9 for i in range(9)] #Create an empty board
    num_empty_cells = 0 
    for y, row in enumerate(board): #for the given board, count the number of 
        if len(row) != 9: #check for any rows that are the wrong length, and raise an error
            print("Error, Input row of incorrect length")
        for x, cell in enumerate(row):
            if cell is None or cell is ['0']:
                num_empty_cells += 1
                continue
            board_copy[y][x] = int(cell) #assign the value in the cell to the copy of the board
    if num_empty_cells == 0: #There are no empty cells in the board, it has already been solved
        return board_copy #Return solution
    possibilities = [[val+1] for val in range(9)]#Records all possible values for a cell
    #Begin Parrallelization 
    with Pool(num_processors) as process: #Create Pool and Processors with given count
        while True:
            for attempt in possibilities:
                if len(attempt) > num_empty_cells: # The attempt created is too long for the 
                    del attempt
            if len(possibilities) == 0: #The program has run out of possible solutions for the board
                raise Exception("No solutions found!!")
            tests = possibilities #give the possible solutuions to be tested 
            succesful_boards = process.starmap(board_verification, ((board_copy, attempt) for attempt in tests)) # Multiple processors use the board verification method
            possibilities = [] #Empty possibilities to prep for next iterations
            for test_num, success in list(enumerate(succesful_boards)):
                if success:
                    if len(tests[test_num]) == num_empty_cells:
                        for y, row in enumerate(board_copy):
                            for x, cell in enumerate(row):
                                if cell is None: #Empty Cell
                                    board_copy[y][x] = tests[test_num][0] #append the active board with test value
                                    del tests[test_num][0] #Remove test to avoid repetition 
                        return board_copy #Return Solution
                    for j in range(9):
                        possibilities.append(tests[test_num] + [j+1]) #add new tests to the possibilities

"""
Goal: Verify a solution
Arguments: A 9x9 Sudoku Board, and number of processors to be used
Return: The solution to the initial given board
"""
def board_verification(board, attempt):
    col = 0
    row = 0
    active_board = [list(row) for row in board] # Copy the argument into this active board
    for value in attempt:
        while active_board[row][col] is not None:
            col += 1 #Move to the next space to the left
            if col >= 9: #Passed the end of the row, continue to the next row
                col = 0 #Reset the column value 
                row += 1 #Increment the row value to 
                if row >= 9: # Check for exceeding the board 
                    break
        if row >= 9: #Exceeds the length of the board
            break
        if value in active_board[row]: #Value exists in another cell within this row
            return False
        if value in (x[col] for x in active_board): #Value exists in another cell within this column
            return False
        for box_row in range((row // 3) * 3, (row // 3) * 3 + 3): #Value exists in another cell within this Block
            for box_col in range((col // 3) * 3, (col // 3) * 3 + 3):
                if active_board[box_row][box_col] == value:
                    return False
        active_board[row][col] = value # Add active value in the solution, progress has been made
    return True # The board is a possible solution


#Processing begins here
if __name__ == '__main__':
    input_file=sys.argv[1] #retrieve the file the puzzle originates from 
    num_processors = int(sys.argv[2]) #
    if(num_processors<1 or num_processors>32): #Ensures processor count is not too high or too low
        print("Improper number of threads")
        sys.exit(0)
    board = loadBoard(input_file) #create a list of lists to represent the board
    knowns =0
    for x in board: #Calculate number of known values
        for y in x:
            if(y is not None):
                knowns+=1
    print("Starting with knowns:", knowns)
    print("Processing.....\n")
    start_time = datetime.now() # Begin Timer 
    solved_board = algorithm(board, num_processors) #Solving Board
    elapsed_time =datetime.now()-start_time
    print("Time Elapsed During Processing: ",str(elapsed_time))
    print("The Solution for the Board is: ")
    printBoard(solved_board)
   
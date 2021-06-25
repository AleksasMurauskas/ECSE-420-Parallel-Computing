import math
from datetime import datetime
import sys

class BruteForceSolver():
  def __init__(self):
    self.boardDim = 9
    self.blockDim=3
    self.board =[]
    self.total_guesses = 0
    self.knowns =[]
  
  def loadBoard(self, file):
    f= open(file,'r')
    inputString= f.read()
    self.board = []
    self.knowns=[]
    print(inputString)
    split_input = inputString.split("\n")
    print(split_input)
    for y, row in enumerate(split_input):
      for x, val in enumerate(row):
        num_val = int(val)
        self.board.append(num_val) # Add Value to Board 
        if int(val) !=0:
          self.knowns.append((9*y+x)) # Mark the position on the known values


  def printBoard(self):
    printString = "".join(['\n' + str(val) if value_loc % self.boardDim == 0 else str(val) for value_loc, val in enumerate(self.board)])
    print(printString)


  def checkBlockValidity(self, active_pos, attempt):
    y = int(math.floor(active_pos/9))
    x =active_pos%9
    block_y = int(math.floor(y/3))
    block_x = int(math.floor(x/3))
    x_range = block_x*3
    y_range = block_y*3
    x_range2 = x_range +2
    y_range2 = y_range +2

    for a in range(y_range, y_range2+1):
      for b in range(x_range, x_range2+1):
        comp_loc = b+(a*9)
        if self.board[comp_loc] == attempt:
          return False
    return True


  def checkRowValidity(self, active_pos, attempt):
    y = int(math.floor(active_pos/9))
    range1 = y*9
    for comp_loc in range(range1, range1+9):
      if(comp_loc!=active_pos and self.board[comp_loc]==attempt):
        return False
    return True


  def checkColumnValidity(self, active_pos, attempt):
    x = active_pos%9
    for a in range(0,9):
      comp_loc = x + (9*a)
      if comp_loc != active_pos and self.board[comp_loc] == attempt:
        return False
    return True


  def checkValidity(self, active_pos, attempt):
    if self.checkColumnValidity(active_pos,attempt) and self.checkRowValidity(active_pos,attempt) and self.checkBlockValidity(active_pos,attempt):
      return True
    else:
      return False


  def runBruteForce(self):
    self.total_guesses = 0
    attemptList = self.algorithm(0,1)
    while attemptList is not None:
      attemptList = self.algorithm(attemptList[0],attemptList[1])
      #print(str(self.total_guesses) + "\n")



  def resetBoard(self, active_pos):
    for a in range(active_pos, len(self.board)):
      if a not in self.knowns:
        self.board[a]=0


  def algorithm(self, active_pos, attempt_start):
    if(active_pos<0 or active_pos>len(self.board)):
      raise Exception("Invalid puzzle index" + str(active_pos))
    previous_valid_attempt =None
    current_attempt_valid = False
    for x in range(active_pos, len(self.board)):
      if x not in self.knowns:
        current_attempt_valid = False
        for attempt in range(attempt_start, self.boardDim+1):
          self.total_guesses +=1
          if self.checkValidity(x,attempt):
            current_attempt_valid=True
            previous_valid_attempt=x
            self.board[x]=attempt
            break
        attempt_start=1
        if not current_attempt_valid:
          break
    if not current_attempt_valid:
      active_pos2 = previous_valid_attempt if previous_valid_attempt is not None else active_pos-1
      attempt_start2 = self.board[active_pos2]+1
      self.resetBoard(active_pos2)
      while (attempt_start2>9 or active_pos2 in self.knowns):
        active_pos2 -=1
        attempt_start2 = self.board[active_pos2]+1
        self.resetBoard(active_pos2)
      return (active_pos2,attempt_start2)
    else:
      return None



#Read File 
input_file=sys.argv[1]
solution =  BruteForceSolver()
solution.loadBoard(sys.argv[1])

print("The Loaded Puzzle looks like this:")
solution.printBoard()

print("Starting with knowns:", len(solution.knowns))

print("Processing")
start_time = datetime.now()

solution.runBruteForce()
elapsed_time =datetime.now()-start_time
print("Time Elapsed During Processing: ",str(elapsed_time))

print("The Solution for the board is:")
solution.printBoard()






#include <iostream>
#include <string>
#include <fstream>
#include <thread>
#include <vector>
#include <stack>
#include <algorithm>
#include <unordered_map>
#include <time.h>
#include <string>
#include <chrono>
using namespace std::chrono;
using namespace std;

const int boardSize = 9;
int box_size = 3;
int depth;
int threadNum;
unordered_map<int, vector<vector<vector<int> > > > combinations;


//this is a sudoku class that will hold the board, markup and other methods to set and find methods we need for the
//crooks algorithm
class Sudoku {
public:
    unsigned markup[boardSize][boardSize]; //will hold the markup of cells
    int board[boardSize][boardSize]; // will hold the values solved
    Sudoku() {};

    void printPuzzle() { //show what the board looks like
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++)
                cout << board[i][j] << " ";
            cout << endl;
        }
    }

    //returns true if markup contains value
    bool markupContains(int i, int j, int value) {
        return (markup[i][j] >> (value-1)) & 1;
    }
    //removes a value from the current markup array
    bool removeFromMarkup(int i, int j, int value) {
        bool ret = markupContains(i, j, value);
        markup[i][j] &= ~(1 << (value-1));
        return ret;
    }

    // init the markup array based on what is available in the board from the crooks algorithm
    void initMarkup() {
        unsigned x = 0;
        for (int k = 0; k < boardSize; k++)
            x |= 1 << k;
        for (int i = 0; i < boardSize; i++)
            for (int j = 0; j < boardSize; j++)
                markup[i][j] = board[i][j]? 0:x; //set the markup based on the board
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                if (board[i][j]) {
                    for (int k = 0; k < boardSize; k++) {
                        removeFromMarkup(i, k, board[i][j]);
                        removeFromMarkup(k, j, board[i][j]);
                        int box_x = k/box_size+i/box_size*box_size;
                        int box_y = k%box_size+j/box_size*box_size;
                        removeFromMarkup(box_x, box_y, board[i][j]);
                    }
                }
            }
        }
    }

    //checks how many cells have 0, which means that don't have a value yet
    int getEmptyCellNum() {
        int num = 0;
        for (int i = 0; i < boardSize; i++)
            for (int j = 0; j < boardSize; j++)
               if (board[i][j] == 0) num++;
        return num;
    }

    // used to find preemtive sets, get the index of unfilled cells based on row column or block
    vector<int> getUnfilledIndex(int caseNum, int i) {
        //use index to find the value for the row or column that is unfilled and push it in the vector to return
        vector<int> unfilled;

        for (int k = 0; k < boardSize; k++) { //need the other values 0 to 8 for the index to accompany i based on the
            //case of needing a row or column
            if (caseNum == 0) { //for rows
                if (!board[i][k]) unfilled.push_back(k);
            }
            else if (caseNum == 1) { //for column
                if (!board[k][i]) unfilled.push_back(k);
            }
            else if (caseNum == 2) { // for box
                if (!board[box_size*i/box_size+k/box_size][box_size*i%box_size+k%box_size]) unfilled.push_back(k);
            }
        }
        return unfilled;
    }

    // update the board with a new value and update the markup array accordingly
    void setBoardVal(int row, int column, int val) {
        board[row][column] = val; //update the value of the board at location (row, column) with val
        markup[row][column] = 0; // set markup at same location at 0 since found

        //go through indexes 0 to 8
        for (int i = 0; i < boardSize; i++) {
            //remove from markup using row
            removeFromMarkup(row, i, val);
            //remove from markup using column
            removeFromMarkup(i, column, val);
            int box_x = i/box_size+row/box_size*box_size;
            int box_y = i%box_size+column/box_size*box_size;
            //remove from markup using the box
            removeFromMarkup(box_x, box_y, val);
            //these 3 ensure the markup is update correctly for future ops
        }
    }

    // the method that runs elimination to check if a cell has only one possible value and set it to that
    bool elimination(){
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                int x = 0;
                int idx = 0;
                for (int k = 0; k < boardSize; k++) {
                    if (markup[i][j] >> k) {
                        x++;
                        idx = k;
                    }
                }
                if (x == 1) {
                    setBoardVal(i, j, idx + 1);
                }
            }
        }
        int emptyCellsLeft = getEmptyCellNum();
        if(emptyCellsLeft == 0){
            return true; //true for no cells left, the puzzle is solved
        }
        return false;
    }

    //the method that takes care of long rangers (
    bool singleton(){
        // check rows
        for (int i = 0; i < boardSize; i++) {
            for (int k = 1; k <= boardSize; k++) {
                int found = 0;
                int col = -1;
                for (int j = 0; j < boardSize; j++) {
                    if (markupContains(i, j, k)) {
                        found++;
                        col = j; //column index
                    }
                }
                if (found == 1) { //found singleton for row, update val
                    setBoardVal(i, col, k);
                    return true;
                }
            }
        }

        // check column
        for (int j = 0; j < boardSize; j++) {
            for (int k = 1; k <= boardSize; k++) {
                int found = 0;
                int row = -1;
                for (int i = 0; i < boardSize; i++) {
                    if (markupContains(i, j, k)) {
                        found++;
                        row = i; //row index
                        if (found > 1)    break;
                    }
                }
                if (found == 1) { //found singleton for column, update val
                    setBoardVal(row, j, k);
                    return true;
                }
            }
        }


        // check box
        for (int i = 0; i < boardSize; i+=box_size) {
            for (int j = 0; j < boardSize; j+=box_size) {
                for (int k = 1; k <= boardSize; k++) {
                    int found = 0;
                    int idx = -1;
                    for (int x = 0; x < box_size; x++) {
                        for (int y = 0; y < box_size; y++) {
                            if (markupContains(i+x, j+y, k)) {
                                found++;
                                idx = x * box_size + y; //compute index of box
                                if (found > 1) break;
                            }
                        }
                        if (found > 1)  break;
                    }
                    if (found == 1) { //found singleton in box, update val
                        setBoardVal(idx/box_size+i, idx%box_size+j, k); //set index based on idx found
                        return true;
                    }
                }
            }
        }
        return false;
    }

    //method to find the preemtive set described by crookes
    bool findPreemptiveSet(int setSize){
        //---------------- check for row ------------------------------
        for (int row = 0; row < boardSize; row++) {
            // get empty cells using the case 0 for rows
            vector<int> emptyCells = getUnfilledIndex(0, row);
            int emptyNum = (int) emptyCells.size();
            if (emptyNum < setSize)  continue;
            vector<vector<int>> idxs = combinations[emptyNum][setSize-1];
            for (int i = 0; i < idxs.size(); i++) {
                int result = 0;
                vector<int> tempVect;

                for (int idx : idxs[i]) { //add in temp vect
                    tempVect.push_back(emptyCells[idx]);
                    result |= markup[row][emptyCells[idx]]; //or to get markup value;
                }
                int counter = 0;
                for (int k = 0; k < boardSize; k++) {
                    if ((result >> k) & 1) counter++; //count to check if it is preemptive set or no
                }
                if (counter == setSize) { //if the amount is the same as the set size
                    // it means we found preemptive set
                    bool isOK = false;
                    for (int k = 0; k < boardSize; k++) {
                        if ((result >> k) & 1) {
                            for (int y = 0; y < boardSize; y++)
                                if (find(tempVect.begin(), tempVect.end(), y) == tempVect.end()) //if found
                                    if (removeFromMarkup(row, y, k+1)) //remove from markup and return true since found
                                        isOK = true;
                        }
                    }
                    return isOK;
                }
            }


            //---------------- check for column ------------------------------
            for (int col = 0; col < boardSize; col++) {
                // choose set of setSize cells
                vector<int> emptyCells = getUnfilledIndex(1, col);
                int emptyNum = (int) emptyCells.size();
                if (emptyNum < setSize)  continue;
                vector<vector<int>> idxs = combinations[emptyNum][setSize-1];
                for (int i = 0; i < idxs.size(); i++) {
                    int result = 0;
                    vector<int> tempVect;
                    for (int idx : idxs[i]) {
                        tempVect.push_back(emptyCells[idx]);
                        result |= markup[emptyCells[idx]][col];
                    }
                    int counter = 0;
                    for (int k = 0; k < boardSize; k++) {
                        if ((result >> k) & 1) counter++; //count to check if it is preemptive set or no
                    }
                    if (counter == setSize) { //if the amount is the same as the set size
                        // it means we found preemptive set
                        bool isOK = false;
                        for (int k = 0; k < boardSize; k++) {
                            if ((result >> k) & 1) {
                                for (int x = 0; x < boardSize; x++)
                                    if (find(tempVect.begin(), tempVect.end(), x) == tempVect.end()) //if found
                                        if (removeFromMarkup(x, col, k+1)) //we can remove from markup and return true
                                            isOK = true;
                            }
                        }
                        return isOK;
                    }
                }
            }
        }
        return false; //return false otherwise as preemtive set was not found
    }
};

vector<vector<int>> findCombinations(int x, int y)
{
    vector<bool> vect(x);
    vector<vector<int>> combinations; // will hold possible
    fill(vect.begin(), vect.begin() + y, true); //fill bool vector

    do {
        vector<int> combo;
        for (int i = 0; i < x; ++i) { //all the values upto input x
            if (vect[i]) { //if the vector is true
                combo.push_back(i); //push back the value to the combo vector
            }
        }
        combinations.push_back(combo); //add this to a vector of vectors for each cell
    } while (prev_permutation(vect.begin(), vect.end())); //while still arrangements
    return combinations; //return this to use with the combinations map described above
}

//check if the value appears in the row, column or box and if it is there return true else return false
bool valueExists(int sudokuPuzzle[boardSize][boardSize], int x, int y, int val) {

    //row
    for (int i = 0; i < boardSize; i++) {
        if (sudokuPuzzle[i][y] == val)
            return true;
    }

    //column
    for (int j = 0; j < boardSize; j++) {
        if (sudokuPuzzle[x][j] == val)
            return true;
    }

    //box
    for (int i = 0; i < box_size; i++) {
        for (int j = 0; j < box_size; j++) {
            if (sudokuPuzzle[(x/box_size)*box_size + i][(y/box_size)*box_size + j] == val)
                return true;
        }
    }

    return false;
}

//find the next index with no value in the board
int findNextEmptyCellIndex(int sudokuPuzzle[boardSize][boardSize], int idx) {
    for (int i = idx; i < boardSize*boardSize; i++) {
        if (sudokuPuzzle[i/boardSize][i%boardSize] == 0) {
            return i;
        }
    }
    return -1; //not found
}

//this is the utility used to queue in backtracking
bool backtrackingQueue(deque<pair<int, Sudoku>> &queue, Sudoku sudokuPuzzle, int idx, int depth) {
    int i = idx/boardSize; //from index number get i and j
    int j = idx%boardSize;

    if (idx == -1) { //empty cell index was not found
        queue.push_back(pair<int, Sudoku>(idx, sudokuPuzzle));
        return true;
    }

    if (!depth) {
        queue.push_back(pair<int, Sudoku>(idx, sudokuPuzzle));
        return false;
    }

    for (int k = 1; k <= boardSize; k++) {
        if (!valueExists(sudokuPuzzle.board, i, j, k)) {
            sudokuPuzzle.board[i][j] = k;
            if (backtrackingQueue(queue, sudokuPuzzle, findNextEmptyCellIndex(sudokuPuzzle.board, idx + 1), depth - 1))
                return true;
        }
    }

    return false;
}

//utility used to queue in backtracking
bool backtrackingStack(stack<pair<int, Sudoku>> &stack, Sudoku sudokuPuzzle, int idx, int depth) {
    int i = idx/boardSize; //from index number get i and j
    int j = idx%boardSize;

    if (idx == -1) { //if the id was not found
        stack.push(pair<int, Sudoku>(idx, sudokuPuzzle));
        return true;
    }

    if (!depth) {
        stack.push(pair<int, Sudoku>(idx, sudokuPuzzle));
        return false;
    }

    for (int k = 1; k <= boardSize; k++) {
        if (!valueExists(sudokuPuzzle.board, i, j, k)) {
            sudokuPuzzle.board[i][j] = k;
            if (backtrackingStack(stack, sudokuPuzzle, findNextEmptyCellIndex(sudokuPuzzle.board, idx + 1), depth - 1))
                return true;
        }
    }

    return false;
}

//backtracking algorithm for the sudoku
void backtracking(Sudoku &sudokuFromCrook) {
    stack<pair<int, Sudoku>> Stack;
    deque<pair<int, Sudoku>> v;
    vector<thread> threads;

    Sudoku tmp(sudokuFromCrook);

    bool ok = false;
    int idx = findNextEmptyCellIndex(tmp.board, 0); //index for backtracking queue
    backtrackingQueue(v, tmp, idx, depth);


    for (int id = 0; id < threadNum; id++) {
            threads.push_back(thread([&ok, &v, id, &sudokuFromCrook](){

                int threadLoad = (int)v.size()/threadNum;
                int endPoint;
                if(id == threadNum){ //figure end point
                    endPoint = v.size();
                } else {
                    endPoint = (1+id)*threadLoad;
                }
                stack<pair<int, Sudoku>> Stack(deque<pair<int, Sudoku>>(v.begin()+id*threadLoad, v.begin()+endPoint));

                for (int i = id*threadLoad; i < (id+1)*threadLoad; i++) {
                    Stack.push(v[i]);
                    if (id == threadNum-1 && (v.size() % threadNum)) {
                        for (int j = (id+1)*threadLoad; j<v.size();j++) {
                            Stack.push(v[j]);
                        }
                    }
                }

                while (!ok) {
                    if (Stack.size()) { //has something
                        int index = Stack.top().first; // get first as index
                        Sudoku puzzle = Stack.top().second; //set sudoku as second in stack
                        Stack.pop(); //pop
                        if (puzzle.getEmptyCellNum() == 0) {
                            sudokuFromCrook = puzzle;
                            ok = true;
                            break;
                        }
                        backtrackingStack(Stack, puzzle, index, depth);
                    }
                    else break;
                }

            }));
    }

    for (auto& thread:threads) {
        thread.join();
    }
}

int main(int numArgs, char* args[]) {
    Sudoku sudokuPuzzle; //init sudoku class
    string fileName;
    depth = 10; //for backtrack if necessary
    threadNum = 1;
    if (numArgs < 2) {
        printf("No file name entered\n");
        return 1;
    }
    else {
        fileName = args[1];
        if(numArgs >= 3){
            threadNum = atoi(args[2]);
        }
    }
    //here number of threads and depth is defined


    // get combinations and put in map
    for (int i = 1; i <= 9; i++)
        for (int j = 1; j <= i; j++)
            combinations[i].push_back(findCombinations(i, j));

    // read from file input and parse into array the inputs of puzzle
    ifstream sudokuFile (fileName); //absolute path


    std::string str;
    for (int i = 0; i < boardSize; i++) {
        std::getline(sudokuFile, str);
        for (int j = 0; j < boardSize; j++) {
            if(str[j] == '\n') break;
            sudokuPuzzle.board[i][j] = str[j] - '0';
        }
    }

    cout<<"\nINITIAL READ SUDOKU PUZZLE: \n";
    sudokuPuzzle.printPuzzle();


    Sudoku solved;
    bool ok = false;
    bool change = false;
    solved = sudokuPuzzle;

    //start time here
    auto start = high_resolution_clock::now();

    solved.initMarkup();

    //run crooks algorithm
    while (!ok){
        //step 1 elimination
        ok = solved.elimination();
        if (ok) break;
        change = false;


        //step 2 singleton
        if (solved.singleton()) { // if we observe changes we restart with elimination
            continue;
        } else {
            ok = false;
        }

        //step 3 preemptive sets
        for (int i = 2; i < boardSize; i++) {
            if (solved.findPreemptiveSet(i)) { // if we observe changes we restart with elimination
                change = true;
                break;
            }
        }
        if (!change) break; // no progress, leave it to backtracking
    }

    //crooks did not do the job therefore we need to call the backtracking algorithm that is threaded
    if (!ok) {
        backtracking(solved);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout<<"\nSOLVED SUDOKU PUZZLE:\n";
    solved.printPuzzle();

    cout<<"\nThreads used: "<<threadNum<<" -----  Time taken to solve is: "<<duration.count()<<" microseconds\n";


    return 0;
}
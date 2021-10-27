# -*- coding: utf-8 -*-
''' This is my player which (almost) always win (hopefully).
'''

import Goban
from playerInterface import *

import numpy as np
import matplotlib.pyplot as plt
import random


class Node:
    def __init__(self, parent=None):
        self.parent = parent
        self.is_leaf = True
        self.children = []
        self.p = 0
        self.n = 0
        self.w = 0

    def q(self):
        pass

    def u(self):
        pass

    def choose_children(self):
        max_value = 0
        for child in self.children:
            child_value = child.q() + child.u()
            if child_value > max_value:
                max_value = child_value
        return max_value

    def expand(self): # 1 simulation par fils ; remonter quand tout 1 niveau est simuler
        #if self.is_leaf:

        pass


def chooseRandomMove(board, color):
        moves = list(board._empties)
        moves.append(-1)
        while moves != []:
            move = random.choice(moves)
            del moves[moves.index(move)]
            if move == -1 or not board._is_suicide(move, color) and not board._is_super_ko(move, color)[0]:
                return move
        return None

def rollout(board, mycolor):
    nb_move_played = 0
    while not board.is_game_over():
        #moves = board.legal_moves()
        #random_move = moves[random.randrange(len(moves))] # random.choice ?
        random_move = chooseRandomMove(board, mycolor)

        board.push(random_move)
        nb_move_played += 1

    # who won ? (we won = 1, we lost = 0)
    result = int(board.final_go_score()[0].lower() == Goban.Board.player_name(mycolor)[0]) #1:win, 0:lose
    
    for _ in range(nb_move_played):
        board.pop()

    return result

simulations = 20

def getValue(board, mycolor, move):
    board.push(move)

    total = 0
    for i in range(simulations):
        total += rollout(board, mycolor) #1 si gagne, 0 sinon

    board.pop()

    return total/simulations



class myPlayer(PlayerInterface):

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None

    def getPlayerName(self):
        return "My Player"

    def getPlayerMove(self):
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS"

        # Get the list of all possible moves
        moves = self._board.legal_moves() # Dont use weak_legal_moves() here!

        best_value = 0
        best_move = moves[0]
        for move in moves:
            value = getValue(self._board, self._mycolor, move) ###
            if value > best_value:
                best_value = value
                best_move = move

        self._board.push(best_move)

        # Let's plot some board probabilities
        #import go_plot
        # We plot probs
        #go_plot.plot_play_probabilities(self._board, probabilities)
        #plt.show()

        #move = np.random.choice(range(82), p=probabilities)
        # Correct number for PASS
        #if move == 81:
        #    move = -1
        #self._board.push(move)

        #int(self._board.final_go_score()[0].lower() == Goban.Board.player_name(self._mycolor)[0]) #1:win, 0:lose

        # New here: allows to consider internal representations of moves
        #print("I am playing ", self._board.move_to_str(move))
        #print("My current board :")
        #self._board.prettyPrint()

        # move is an internal representation. To communicate with the interface I need to change it to a string
        return Goban.Board.flat_to_name(best_move)

    def playOpponentMove(self, move):
        #print("Opponent played ", move, "i.e. ", move) # New here
        # the board needs an internal represetation to push the move.  Not a string
        self._board.push(Goban.Board.name_to_flat(move))

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")


# modifier : 
#play oponnent move
#new game
#get player move
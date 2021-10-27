# -*- coding: utf-8 -*-
''' This is the famous random player which (almost) always looses.
'''

from numpy.core.fromnumeric import shape
import Goban
from playerInterface import *

import numpy as np
import matplotlib.pyplot as plt

import random

class myPlayer(PlayerInterface):
    ''' Example of a random player for the go. The only tricky part is to be able to handle
    the internal representation of moves given by legal_moves() and used by push() and
    to translate them to the GO-move strings "A1", ..., "J8", "PASS". Easy!

    '''

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None

    def getPlayerName(self):
        return "Random Player"

    def getPlayerMove(self):
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS"
        # Get the list of all possible moves
        moves = self._board.legal_moves() # Dont use weak_legal_moves() here!

        # Let's plot some board probabilities
        import go_plot
        # Generate random proabibilities
        probabilities = np.zeros(self._board._BOARDSIZE ** 2 + 1)
        mcts = self.MCTS(20)
        for i in range(len(mcts)):
            probabilities[int(mcts[i,0])] = mcts[i,1]
        # Now we want to to put to 0 all impossible moves
        # SO we careate multiplier with 0 everywhere and put 1 where the move is legal
        multiplier = np.zeros_like(probabilities)
        for some_move in moves:
            x, y = Goban.Board.unflatten(some_move)
            if x == y == -1:
                continue
            multiplier[y * Goban.Board._BOARDSIZE + x] = 1
        # Pass move is always legal
        multiplier[-1] = 1
        # Now we multiply our probs
        probabilities *= multiplier

        
        sum = np.sum(probabilities)
        if sum == 0:
            moves = self._board.legal_moves()
            # Generate random proabibilities
            probabilities = np.random.uniform(size=self._board._BOARDSIZE ** 2 + 1)
            # Now we want to to put to 0 all impossible moves
            # SO we careate multiplier with 0 everywhere and put 1 where the move is legal
            multiplier = np.zeros_like(probabilities)
            for some_move in moves:
                x, y = Goban.Board.unflatten(some_move)
                if x == y == -1:
                    continue 
                multiplier[y * Goban.Board._BOARDSIZE + x] = 1
            # Pass move is always legal
            multiplier[-1] = 1
            # Now we multiply our probs
            probabilities *= multiplier

        # Normalize them
        probabilities /= np.sum(probabilities)

        # We plot them
        #go_plot.plot_play_probabilities(self._board, probabilities)

        
        print(probabilities)
        '''
        bestMoves = [-1]
        value = probabilities[81]
        for i in range(81):
            if probabilities[i] > value:
                value = probabilities[i]
                bestMoves = [i]
            elif probabilities[i] == value:
                bestMoves.append(i)
        move = np.random.choice(bestMoves)
        '''
        move = np.random.choice(range(self._board._BOARDSIZE ** 2 + 1), p=probabilities)
        # Correct number for PASS
        if move == self._board._BOARDSIZE ** 2:
            move = -1
        self._board.push(move)

        # New here: allows to consider internal representations of moves
        #print("I am playing ", self._board.move_to_str(move))
        #print("My current board :")
        #self._board.prettyPrint()

        # move is an internal representation. To communicate with the interface I need to change if to a string
        return Goban.Board.flat_to_name(move)

    def MCTS(self, nbGames):
        legals = self._board.legal_moves()
        proba = []
        for move in legals:
            proba.append(self.probability(move, nbGames))
        return np.array(proba)

    def probability(self, move, nbGames):
        self._board.push(move)
        victories = 0
        for i in range(nbGames):
            if self.rollout():
                victories += 1
        self._board.pop()
        return (move, victories / nbGames)

    def rollout(self):
        i = 0
        color = self._mycolor
        while not self._board._gameOver:
            move = self.chooseRandomMove(color)
            self._board.push(move)
            i += 1
            if color == self._board._WHITE:
                color = self._board._BLACK
            else:
                color = self._board._WHITE
        
        result = self._board.result()
        
        test = False

        if result == "1-0" and self._mycolor == self._board._WHITE or result == "0-1" and self._mycolor == self._board._BLACK:
            test = True

        while i > 0:
            self._board.pop()
            i -= 1

        return test

    def chooseRandomMove(self, color):
        moves = list(self._board._empties)
        moves.append(-1)
        while moves != []:
            move = random.choice(moves)
            del moves[moves.index(move)]
            if move == -1 or not self._board._is_suicide(move, color) and not self._board._is_super_ko(move, color)[0]:
                return move
        return None

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

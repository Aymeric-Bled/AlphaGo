# -*- coding: utf-8 -*-
''' This is my player which (almost) always win (hopefully).
'''

import Goban
from playerInterface import *

import numpy as np
import matplotlib.pyplot as plt
import random
import math

C_PUCT = 10 # facteur d'exploration


class Node:
    def __init__(self, mycolor, move, parent=None):
        self.parent = parent
        self.is_leaf = True
        self.children = []
        self.p = 1
        self.n = 0
        self.w = 0
        self.mycolor = mycolor
        self.move = move
        # stocker boards ou moves ?

    def q(self):
        if self.n == 0:
            return 0
        return self.w / self.n

    def u(self):
        n_sum = 0
        for child in self.children:
            n_sum += child.n

        return self.p * math.sqrt(self.n) / (1 + self.n) * C_PUCT

    def choose_children(self):
        max_value = -1
        best_children = []
        for child in self.children:
            child_value = child.q() + child.u()
            if child_value > max_value:
                max_value = child_value
                best_children = [child]
            elif child_value == max_value:
                best_children.append(child)
        return random.choice(best_children)

    def expand(self, board): # 1 simulation par fils ; remonter quand tout 1 niveau est simuler ; repartir d'en haut
        if self.is_leaf:
            if board.is_game_over():
                node = self
                result = board.final_go_score()[0]
                while node != None:
                    if result == "B" and node.mycolor == board._WHITE or result == "W" and node.mycolor == board._BLACK:
                        node.w += 1
                    node.n += 1
                    node = node.parent
                return
            moves = board.legal_moves()
            if moves != []:
                self.is_leaf = False
                for move in moves:
                    board.push(move)
                    if self.mycolor == board._BLACK:
                        newcolor = board._WHITE
                    else:
                        newcolor = board._BLACK
                    node = Node(newcolor, move, self)
                    self.children.append(node)
                    result = rollout(board, newcolor)
                    board.pop()
                    while node != None:
                        if result == "B" and node.mycolor == board._WHITE or result == "W" and node.mycolor == board._BLACK:
                            node.w += 1
                        node.n += 1
                        node = node.parent
        else:
            children = self.choose_children()
            board.push(children.move)
            children.expand(board)
            board.pop()




def chooseRandomMove(board, color):
    moves = list(board._empties)
    moves.append(-1)
    length = len(moves)
    while moves != []:
        i = random.randint(0, length - 1)
        move = moves[i]
        del moves[i]
        length -= 1
        if move == -1 or not board._is_suicide(move, color) and not board._is_super_ko(move, color)[0]:
            return move
    return None

def rollout(board, mycolor):
    nb_move_played = 0
    while not board.is_game_over():
        random_move = chooseRandomMove(board, mycolor)
        board.push(random_move)
        nb_move_played += 1
    # who won ? (we won = 1, we lost = 0)
    result = board.final_go_score()[0] #1:win, 0:lose
    for _ in range(nb_move_played):
        board.pop()
    return result

simulations = 20

def getValue(board, mycolor, move):
    board.push(move)
    total = 0
    for _ in range(simulations):
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
        import time
        endtime = time.time() + 3
        end = False
        while time.time() < endtime:
            self._root.expand(self._board)
        max = -1000
        best_moves = []
        for node in self._root.children:
            if node.q() - node.u() > max:
                best_moves = [node.move]
                max = node.q() - node.u()
            elif node.q() - node.u() == max:
                best_moves.append(node.move)
        best_move = random.choice(best_moves)

        self._board.push(best_move)
        children = self._root.children
        self._root = Node(self._opponent, best_move)
        for node in children:
            if node.move == best_move:
                self._root = node
        return Goban.Board.flat_to_name(best_move)

    def playOpponentMove(self, move):
        #print("Opponent played ", move, "i.e. ", move) # New here
        # the board needs an internal represetation to push the move.  Not a string
        move = Goban.Board.name_to_flat(move)
        self._board.push(move)
        children = self._root.children
        self._root = Node(self._mycolor, move)
        for node in children:
            if node.move == move:
                self._root = node

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)
        self._root = Node(Goban.Board._BLACK, None)

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")


# modifier : 
#play oponnent move
#new game
#get player move
# -*- coding: utf-8 -*-
''' This is my player which (almost) always win (hopefully).
'''

import Goban
from playerInterface import *

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import CNN
import torch

C_PUCT = 1 # facteur d'exploration


use_nn = True # Set it to False to use only MCTS
if use_nn:
    model = CNN.NeuralNetwork().to('cpu')
    #load weights
    model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))


def board_to_inference(entry_board):
    '''
    format board to be used in model inference
    '''
    board_black_list = []
    board_white_list = []
    board = Goban.Board()

    if len(entry_board._historyMoveNames) < 8:
        for i in range(8 - len(entry_board._historyMoveNames)):
            board_black_list.append(np.zeros(board._BOARDSIZE ** 2, dtype=np.int8))
            board_white_list.append(np.zeros(board._BOARDSIZE ** 2, dtype=np.int8))
    length = len(entry_board._historyMoveNames)
    for index in range(length):
        m = entry_board._historyMoveNames[index]
        if m == 'PASS':
            move = -1
        else:
            (x,y) = CNN.name_to_coord(m)
            move = y * board._BOARDSIZE + x
        board.push(move)
        if length - index <= 8:
            board_black = board._board.copy()
            board_white = board._board.copy()
            for i in range(board._BOARDSIZE ** 2):
                if board_black[i] == 2:
                    board_black[i] = 0
                if board_white[i] == 1:
                    board_white[i] = 0
                if board_white[i] == 2:
                    board_white[i] = 1 
            board_black_list.append(np.array(board_black))
            board_white_list.append(np.array(board_white))
    hist_black = board_black_list[-8:]
    hist_white = board_black_list[-8:]
    hist = [np.array([hist_black[i], hist_white[i]]) for i in range(8)]
    if(length % 2 == 0):
        hist.append(np.array([np.ones(board._BOARDSIZE ** 2, dtype=np.int8), np.ones(board._BOARDSIZE ** 2, dtype=np.int8)])) #BLACK to 1
    else:
        hist.append(np.array([np.zeros(board._BOARDSIZE ** 2, dtype=np.int8), np.zeros(board._BOARDSIZE ** 2, dtype=np.int8)])) #WHITE to 0
    hist = np.array(hist)
    return torch.reshape(torch.from_numpy(hist), (1,9,2,81)).type(torch.float32)


class Node:
    def __init__(self, color, parent=None, move=None):
        self.parent = parent
        self.move = move
        self.is_leaf = True
        self.children = []
        self.p = 1 # prior probability (given by neural network every time we hit a leaf node)
        self.n = 0 # visit count (inc for each node in the path when perform backup)
        self.w = 0 # winning count (update during backup for each node in the path back to root)
        self.color = color

    def q(self):
        if self.n == 0:
            return 0
        return self.w / self.n

    def u(self):
        n_sum = 0
        for child in self.children:
            n_sum += child.n
        return self.p * math.sqrt(n_sum) / (1 + self.n) * C_PUCT

    def getBestChild(self): # select best child (action that maximizes Q+U)
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


class MCTS:
    def __init__(self, board, color):
        self.board = board
        self.mycolor = color
        self.root = Node(board._BLACK)
        self.expand(self.root)

    def select(self):
        current_node = self.root
        while(current_node.is_leaf != True):
            best_child = current_node.getBestChild()
            current_node = best_child
            self.board.push(current_node.move)
        return current_node

    def expand(self, node):
        if self.board.is_game_over():
            #print('Tried to expand while game is over !')
            return None
        if node.is_leaf == False:
            raise ValueError('Tried to expand a node that is not a leaf !')
        else:
            moves = self.board.legal_moves()
            for move in moves:
                color = None
                if node.color == self.board._BLACK:
                    color = self.board._WHITE
                else:
                    color = self.board._BLACK
                child_node = Node(color, node, move)
                node.children.append(child_node)
            node.is_leaf = False
            return node.children

    def backup(self, node, results, rollouts_count):
        while node != self.root.parent: #is not None:
            #if node.depth%2 == 1:
            if node.color != self.mycolor:
                node.w += results
            else:
                node.w += rollouts_count - results
            node.n += rollouts_count
            if node != self.root:
                self.board.pop() # pop moves that we pushed with select
            node = node.parent

    def chooseRandomMove(self, color):
            moves = list(self.board._empties)
            moves.append(-1)
            while moves != []:
                move = random.choice(moves)
                del moves[moves.index(move)]
                if move == -1 or not self.board._is_suicide(move, color) and not self.board._is_super_ko(move, color)[0]:
                    return move
            return None

    def rollout(self):
        """
        Run a rollout and return 1 if we won, 0 otherwise.
        """
        nb_move_played = 0
        while not self.board.is_game_over():
            #moves = board.legal_moves()
            #random_move = moves[random.randrange(len(moves))]
            random_move = self.chooseRandomMove(self.board.next_player())
            self.board.push(random_move)
            nb_move_played += 1
        # who won ? (we won = 1, we lost = 0)
        result = int(self.board.final_go_score()[0].lower() == Goban.Board.player_name(self.mycolor)[0]) #1:win, 0:lose
        for _ in range(nb_move_played):
            self.board.pop()
        return result

    def process(self, seconds):
        t_end = time.time() + seconds
        while time.time() < t_end:

            node = self.select()

            children = self.expand(node)

            # evaluate with rollouts (without neural net)
            if children is not None: # node is not terminal (game is not over)
                # 1 rollout by child ; backup to root when all children got a rollout
                result_sum = 0
                for child in children:
                    self.board.push(child.move)
                    result = self.rollout()
                    self.board.pop()
                    #if node.depth%2 == 0: # children are ours nodes, so store result from our perspective
                    if node.color == self.mycolor:
                        child.w += result
                    else:
                        child.w += 1 - result
                    child.n += 1
                    result_sum += result

                self.backup(node, result_sum, len(children))
            else: # node is terminal (game is over)
                result = self.rollout()
                self.backup(node, result, 1)

    def process_nn(self, seconds):
        t_end = time.time() + seconds
        while time.time() < t_end:

            node = self.select()

            # game state of leaf node passed into neural network which ouput predictions (p:move probs, v:value of the state)
            # p is attached to the new feasible actions from the leaf node
            
            # construire entr??e r??seau ?? partir de node et ses parents...
            x = board_to_inference(self.board)
            p, v = model.forward(x)
            if node.color == self.board._BLACK:
                v = 1 - v

            children = self.expand(node)

            if children is not None:
                v_sum = 0
                for child in children:
                    child.p = p[child.move]
                    if node.color == self.mycolor:
                        child.w += v
                    else:
                        child.w += 1 - v
                    v_sum += v

                self.backup(node, v_sum, len(children))
            else:
                self.backup(node, v, 1)

    def moveToPlay(self):
        tau = 1    # temperature
        if self.board._nbBLACK + self.board._nbWHITE >= self.board._BOARDSIZE:
            tau = 0.1

        length = len(self.root.children)
        n_list = np.zeros(length)
        for i in range(length):
            n_list[i] = self.root.children[i].n ** (1 /tau)

        n_list /= np.sum(n_list)
        
        # pick move from distribution
        child_index = np.random.choice(range(length), p=n_list)
        return self.root.children[child_index].move

    def updateTree(self, move):
        # self.board.push(move) # only if mcts has a deepcopy of the board
        if len(self.root.children) == 0:
            self.expand(self.root)

        new_root = None
        for child in self.root.children:
            if self.board.move_to_str(child.move) == self.board.move_to_str(move):
                new_root = child
                break
        self.root = new_root
        #self.root.parent = None


### Version 1 (simulations on first level)
# simulations = 20
# def getValue(board, mycolor, move):
#     board.push(move)
#     total = 0
#     for i in range(simulations):
#         total += rollout(board, mycolor) #1 si gagne, 0 sinon
#     board.pop()
#     return total/simulations


class myPlayer(PlayerInterface):

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None

    def getPlayerName(self):
        return "Aymeric-Remi"

    def getPlayerMove(self):
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS"

        # Get the list of all possible moves
        #moves = self._board.legal_moves() # Dont use weak_legal_moves() here!

        ### Version 2 (MCTS)
        # update mcts root and state (based on opponent move)
        seconds = 5
        if use_nn:
            self._mcts.process_nn(seconds)
        else:
            self._mcts.process(seconds)
        move = self._mcts.moveToPlay()
        self._mcts.updateTree(move)
        self._board.push(move)

        ### Version 1 (simulations on 1st level)
        # best_value = 0
        # best_move = moves[0]
        # for move in moves:
        #    value = getValue(self._board, self._mycolor, move)
        #    if value > best_value:
        #        best_value = value
        #        best_move = move
        # self._board.push(best_move)


        ### Version 0
        # Let's plot some board probabilities
        # import go_plot
        # We plot probs
        # go_plot.plot_play_probabilities(self._board, probabilities)
        # plt.show()

        # move = np.random.choice(range(82), p=probabilities)
        # Correct number for PASS
        # if move == 81:
        #    move = -1
        # self._board.push(move)

        # int(self._board.final_go_score()[0].lower() == Goban.Board.player_name(self._mycolor)[0]) #1:win, 0:lose

        # New here: allows to consider internal representations of moves
        # print("I am playing ", self._board.move_to_str(move))
        # print("My current board :")
        # self._board.prettyPrint()


        # move is an internal representation. To communicate with the interface I need to change it to a string
        return Goban.Board.flat_to_name(move)

    def playOpponentMove(self, move):
        #print("Opponent played ", move, "i.e. ", move) # New here
        #??the board needs an internal represetation to push the move.  Not a string
        self._mcts.updateTree(Goban.Board.name_to_flat(move))
        self._board.push(Goban.Board.name_to_flat(move))

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)
        self._mcts = MCTS(self._board, color)

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")
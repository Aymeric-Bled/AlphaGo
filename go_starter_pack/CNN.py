from numpy.random.mtrand import randint
import torch
from torch import nn
import numpy as np
import gzip
import json
import os
import urllib

def get_raw_data_go():
    ''' Returns the set of samples from the local file or download it if it does not exists'''

    raw_samples_file = "samples-9x9.json.gz"

    if not os.path.isfile(raw_samples_file):
        print("File", raw_samples_file, "not found, I am downloading it...", end="")
        urllib.request.urlretrieve("https://www.labri.fr/perso/lsimon/ia-inge2/samples-9x9.json.gz", "samples-9x9.json.gz")
        print(" Done")

    with gzip.open("samples-9x9.json.gz") as fz:
        data = json.loads(fz.read().decode("utf-8"))
    return data

class ConvolutionBlock(nn.Module):
    def __init__(self):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(9, 9,kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(9)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        conv = self.conv(x)
        conv += x
        x1 = self.bn(conv)
        return self.relu(x1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.res1 = ConvolutionBlock()
        self.res2 = ConvolutionBlock()
        self.res3 = ConvolutionBlock()
        self.res4 = ConvolutionBlock()
        self.res5 = ConvolutionBlock()
        self.linear1 = nn.Linear(9 * 81 * 2, 82)
        self.linear2 = nn.Linear(9 * 81 * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(0)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = torch.flatten(x)
        p = self.softmax(self.dropout1(self.linear1(x.clone())))
        v = self.sigmoid(self.dropout2(self.linear2(x.clone())))
        
        return (p,v)


#model = NeuralNetwork()

#print(model)
#x = torch.zeros((1,9,81,1))
#result = model(x)

#print(result)

#data = get_raw_data_go()

#print(len(data))

def name_to_coord(s):
    assert s != "PASS"
    indexLetters = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'J':8}

    col = indexLetters[s[0]]
    lin = int(s[1:]) - 1
    return col, lin

def rotation(board):
    L=[]
    L.append(board[:][:])
    for i in range(3):
        rotation = [[board[i][8 - j] for i in range(9)] for j in range(9)]
        board = rotation[:][:]
        L.append(board[:][:])
    return L

def symmetry(board):
    return rotation(board) + rotation([[board[i][j] for i in range(9)] for j in range(9)])

def coord_to_board(black_moves, white_moves):
    board = [[0. for i in range(9)] for j in range(9)]
    for x,y in black_moves:
        board[x][y] = 1.
    for x,y in white_moves:
        board[x][y] = -1.
    return board

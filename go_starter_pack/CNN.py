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

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(9, 9,kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(9, 9,kernel_size=5, padding='same')
        self.conv3 = nn.Conv2d(9, 9,kernel_size=5, padding='same')
        self.conv4 = nn.Conv2d(9, 9,kernel_size=5, padding='same')
        self.conv5 = nn.Conv2d(9, 9,kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm2d(9)
        self.bn2 = nn.BatchNorm2d(9)
        self.bn3 = nn.BatchNorm2d(9)
        self.bn4 = nn.BatchNorm2d(9)
        self.bn5 = nn.BatchNorm2d(9)
        self.linear1 = nn.Linear(9 * 81, 82)
        self.linear2 = nn.Linear(9 * 81, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(0)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        conv = self.relu(self.conv1(x))
        x += conv
        x = self.bn1(x)
        
        conv = self.relu(self.conv2(x))
        x += conv
        x = self.bn2(x)
        
        conv = self.relu(self.conv3(x))
        x += conv
        x = self.bn3(x)

        conv = self.relu(self.conv4(x))
        x += conv
        x = self.bn4(x)
        
        conv = self.relu(self.conv5(x))
        x += conv
        x = self.bn5(x)

        x = torch.flatten(x)
        p = self.softmax(self.dropout(self.linear1(x)))
        v = self.sigmoid(self.dropout(self.linear2(x)))
        
        return (p,v)



model = NeuralNetwork()

print(model)
x = torch.zeros((1,9,81,1))
result = model(x)

print(result)

data = get_raw_data_go()

print(len(data))

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

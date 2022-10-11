
# coding: utf-8
# 五目並べの操作をまとめたモジュール

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import copy


size = 9

# 五目並べのゲーム
class Game:
    # コンストラクタ
    def __init__(self):
        # 盤面のサイズ (size x size)
        self.size = size
        # 盤面情報のリスト
        self.square = [[[0, 0] for _ in range(self.size)] for _ in range(self.size)]
        # 手番(先手黒1, 後手白-1)
        self.turn = 1
        
    # 石の数
    def count(self):
        n = 0
        for r in self.square:
            for c in r:
                if c != [0, 0]:
                    n += 1
        return n

    # (row, column)に石を置く
    def put(self, row, column):
        index = 0 if self.turn is 1 else 1
        if 0 <= row < self.size and 0 <= column < self.size:
            self.square[row][column][index] = 1
        self.turn *= -1
        
    # (row, colomn)に石がおける判定
    def putable(self, row, column):
        if 0 <= row < self.size and 0 <= column < self.size:
            return self.square[row][column] == [0, 0]
        else:
            return 0
        
    # ゲーム終了判定 (1黒勝利, -1白勝利, 0進行中)
    def end_game(self):
        for color in [-1, 1]:
            direction = [[-1, -1], [-1, 0], [-1, 1],
                         [0, -1], [0, 1], [1, -1], [1, 0], [-1, -1]]
            for i in range(self.size):
                for j in range(self.size):
                    for d in direction:
                        if self.fives(color, i, j, d):
                            return color
        return 0

    # 五目連続判定
    def fives(self, color, i, j, d):
        number = 0
        index = 0 if color is 1 else 1
        while 0 <= i < self.size and 0 <= j < self.size and self.square[i][j][index] is 1:
            number += 1
            i += d[0]
            j += d[1]
        if number >= 5:
            return 1
        else:
            return 0
    
    # 盤面を画像で表示
    def iout(self):
        img = np.asarray([[[0, 256, 0] if c == [0, 0] else [0, 0, 0] if c == [1, 0] else [
                         256, 256, 256] for c in r] for r in self.square])
        plt.imshow(img)
        plt.show()

    # 盤面を標準出力
    def aout(self):
        for r in self.square:
            for c in r:
                if c == [0, 0]:
                    print('ー', end='')
                elif c == [1, 0]:
                    print('＊', end='')
                else:
                    print('０', end='')
            print('')
        print('')
    
    # ランダムに石を置く
    def rand_put(self):
        ij = [[i, j] for j in range(self.size) for i in range(self.size) if self.square[i][j] == [0, 0]]
        if len(ij) > 0:
            i, j = ij[rd.randrange(len(ij))]
            self.put(i, j)

    # 次のすべての盤面
    def next_nodes(self):
        n = []
        for i in range(self.size):
            for j in range(self.size):
                if self.putable(i, j):
                    n.append(copy.deepcopy(self))
                    n[-1].put(i, j)
        return n

    # 入力による着手
    def input_put(self):
        i, j = -1, -1
        while (not 0 <= i < self.size) or (not 0 <= j < self.size) or not self.putable(i, j):
            i, j = map(int, input('input (row, column)').split())
        self.put(i, j)


    # 盤面を時計回りに90度回転
    def rotate(self):
        s = copy.deepcopy(self.square)
        for i in range(self.size):
            for j in range(self.size):
                self.square[i][j] = s[j][self.size-i-1]
        
    #盤面を対角線を軸に反転
    def reflect(self):
        s = copy.deepcopy(self.square)
        for i in range(self.size):
            for j in range(self.size):
                self.square[i][j] = s[j][i]


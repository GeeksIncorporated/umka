import csv
import sys
import time

import chess
import torch
from chess.pgn import read_game

pieces_ascii = "KQRBNPkqrbnp"
pieces_unicode = "♚♛♜♝♞♟♔♕♖♗♘♙"
translation_rules = str.maketrans(pieces_ascii, pieces_unicode)

PIECES = {
    ".": 0,
    "P": 1, "N": 3, "B": 3.5, "R": 5, "Q": 9, "K": 0,
    "p": -1, "n": -3, "b": -3.5, "r": -5, "q": -9, "k": 0
}

PIECES_TENSORS = {
    ".": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "P": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "N": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "B": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "R": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "Q": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "K": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "p": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "n": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "b": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "r": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "q": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "k": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}


def show_board(board, material_score, position_score):
    return
    print(str(board).translate(translation_rules))
    print("%.6s %.6s" % (int(material_score), position_score))
    print(str(chess.pgn.Game().from_board(board)).split("\n\n")[1])


def board_tensor(board):
    res = []
    for c in str(board):
        if c == '\n':
            continue
        elif c == ' ':
            continue
        res += PIECES_TENSORS[c]
    return res


def board_material(board):
    """
    Calculates board material in centipawns.
    White advantage positive, black negative.
    :param board:
    :return: score in centipawns
    """
    res = 0
    for c in str(board):
        if c == '\n':
            continue
        elif c == ' ':
            continue
        res += PIECES[c] / 10
    return res


def sample_game(game):
    board = game.board()
    samples = []
    main_line = list(game.main_line())
    for move in main_line:
        if not board.turn:
            board.push(move)
            continue
        position = board_tensor(board=board)
        samples.append(position)
        board.push(move)
    return samples


def load_labels():
    game_labels = {}
    for event_id, position_scores in csv.reader(
            open("data/pgn/labels.csv")):
        game_labels[event_id] = position_scores
    return game_labels


def annotated_sample_generator():
    """
    Generates annotated training data. Uses games.pgn games dataset.
    Each game has an event number. Labels extracted from labels.cs
    Generator yields a pair of samples, labels tensors
    where samples represents game positions like the following:
    [[-0.5,-0.3,-0.35,-0.8,-1.0,-0.35,-0.3,-0.5,
     -0.1,-0.1, -0.1,-0.1,-0.1, -0.1,-0.1,-0.1,
        0,   0,   0,    0,   0,    0,   0,   0,
     ...
     0.1, 0.1,  0.1, 0.1, 0.1,  0.1, 0.1, 0.1,
     0.5, 0.3, 0.35, 0.8, 1.0, 0.35, 0.3, 0.5],[..]..]
    and labels of the form:
    [0.24, 0.18, ..]
    """
    pgn = open("data/pgn/games.pgn")
    game_labels = load_labels()
    while True:
        try:
            game = read_game(pgn)
            if not game:
                break
            event = game.headers["Event"]
            labels = game_labels[event]
            labels = labels.split(' NA')[0]
            labels = list(
                map(lambda x: float(x) / 1000, labels.split(' ')[::2]))
            samples = sample_game(game)
            yield samples[:len(labels)], labels
        except:
            pass


def position_needs_attention(board, move):
    # return False
    return any(
        [board.is_capture(move),
         board.is_castling(move),
         board.is_check()])

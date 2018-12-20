import csv

import chess
from chess.pgn import read_game

pieces_ascii = "KQRBNPkqrbnp"
pieces_unicode = "♚♛♜♝♞♟♔♕♖♗♘♙"
translation_rules = str.maketrans(pieces_ascii, pieces_unicode)

PIECES = {
    ".": 0,
    "P": 1, "N": 3, "B": 3.5, "R": 5, "Q": 9, "K": 0,
    "p": -1, "n": -3, "b": -3.5, "r": -5, "q": -9, "k": 0
}


def show_board(board):
    return
    print(str(board).translate(translation_rules)[::-1])
    print(str(chess.pgn.Game().from_board(board)).split("\n\n")[1])


def board_tensor(board):
    res = []
    for c in str(board):
        if c == '\n':
            continue
        elif c == ' ':
            continue
        res.append(PIECES[c] / 10)
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
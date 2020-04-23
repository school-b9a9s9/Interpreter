from typing import List
import functools
# lexer


# Reading input file
inputFile = open("input.txt", "r")
sourceCode= inputFile.read()

class Token():
    def __init__(self, position):
        self.position = position
    def __repr__(self):
        return "Token"
    def __str__(self):
        return self.position


def lex(sourceCode: str)->List[Token]:

    head, *tail = sourceCode

    # functools.reduce()


    pass


print(lex(sourceCode))
from typing import List, Tuple, Callable, Union

# Reading input file
inputFile = open("input.txt", "r")
sourceCode= inputFile.read()

# ---------------------------------------------
# Classes
# ---------------------------------------------
class Token():
    def __init__(self, type, value, line, position):
        self.type = type
        self.value = value
        self.line = line
        self.position = position
    def __repr__(self):
        return (self.type + " Token")
    def __str__(self):
        return (self.type + ' Token: \'' + self.value + '\' (line: ' + str(self.line) + ', character: ' + str(self.position) + ')')

# ---------------------------------------------
# Lexer
# ---------------------------------------------
# createToken :: str, Tuple[int,int] -> Callable[List[Token]] -> Union[List[Token], None]
def createToken(value: str, position: Tuple[int, int]) -> Union[Callable[[List[Token]], List[Token]], Callable[[List[Token]], None]]:
    def addTokenToList(tokenList):
        tokenList.append(token)
        return tokenList
    def doNothing(tokenList):
        pass

    if len(value) == 0:
        return doNothing
    elif not str.isdigit(value) and not str.isalpha(value):
        token =  Token('TODO', value, position[0], position[1])
    else:
        startPosition = (position[0], position[1] - len(value))
        token = Token('TODO', value, startPosition[0], startPosition[1])
    return addTokenToList

def updateCurrentPosition(currentCharacter):
    def newLine(lastPosition: List[int]):
        newPosition = lastPosition[0] + 1, 1
        return newPosition

    def nextCharacter(lastPosition: List[int]):
        newPosition = lastPosition[0], lastPosition[1] + 1
        return newPosition

    if currentCharacter == '\n':
        return newLine
    else:
        return nextCharacter

# lex :: str -> List[Token]
def lex(sourceCode: str, s: str = '', currentPosition: Tuple[int, int] = (1, 1))->List[Token]: # sourceCode: code, s: not-tokens to be converted a token
    head, *tail = sourceCode
    tokens: List(Token) = []

    if str.isalpha(head) or str.isdigit(head):
        s += head
    else:
        if len(s) > 0:
            createToken(s, currentPosition)(tokens)
        s = ''

        if head != ' ' and head != '\n':
            createToken(head, currentPosition)(tokens)

    currentPosition = updateCurrentPosition(head)(currentPosition)

    if len(tail) > 0:
        if tokens:
            return tokens + lex(tail, s, currentPosition)
        else:
            return lex(tail, s, currentPosition)
    else:
        createToken(s, currentPosition)(tokens)
        return tokens



# ---------------------------------------------
# Run/Debug
# ---------------------------------------------
for i in (lex(sourceCode)):
    print(i)

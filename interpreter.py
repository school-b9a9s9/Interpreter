from typing import List, Tuple, Callable, Union
import re

# Reading input file
inputFile = open("input.txt", "r")
sourceCode= inputFile.read()

# ---------------------------------------------
# Classes
# ---------------------------------------------
class Error():
    def __init__(self, errorMessage: str, line: int, position: int):
        self.errorMessage = errorMessage
        self.line = line
        self.position = position

    def __repr__(self):
        return self.errorMessage + ' at line: ' + str(self.line) + ', position: ' + str(self.position)

class Token():
    def __init__(self, type: str, value: str, line: int, position: int):
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

def getTokenType(value: str) -> str:
    tokenTypes = {
        'ADD'               : r'[+]',
        'MULTIPLY'          : r'[*]',
        'DIVIDE'            : r'[/]',
        'SUBTRACT'          : r'[-]',
        'ASSIGN/COMPARE'    : r'[=]',
        'BLOCK'             : r'[()]',
        'END'               : r';',
        'IF'                : r'^if$',
        'NUMBER'            : r'^[0-9]*$',
        'IDENTIFIER'        : r'\w'             # Has to be last
    }
    for tokenType in tokenTypes:
        if re.match(tokenTypes.get(tokenType), value):
            return tokenType

    return "UNKNOWN"


# --------Vraag: moet ik in deze functie definitie nou iets zeggen over getTokenType of niet?-------
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
        token = Token(getTokenType(value), value, position[0], position[1])
    else:
        startPosition = (position[0], position[1] - len(value))
        token = Token(getTokenType(value), value, startPosition[0], startPosition[1])
    return addTokenToList

def updateCurrentPosition(currentCharacter) -> Callable[[Tuple[int, int]], Tuple[int, int]]:
    def newLine(lastPosition: Tuple[int, int]) -> Tuple[int, int]:
        newPosition = lastPosition[0] + 1, 1
        return newPosition

    def nextCharacter(lastPosition: Tuple[int, int]) -> Tuple[int, int]:
        newPosition = lastPosition[0], lastPosition[1] + 1
        return newPosition

    if currentCharacter == '\n':
        return newLine
    else:
        return nextCharacter

# lex :: str -> List[Token]
def lex(sourceCode: str, s: str = '', currentPosition: Tuple[int, int] = (1, 1)) -> List[Token]: # sourceCode: code, s: not-tokens to be converted a token
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
# Parser
# ---------------------------------------------
PRECEDENCE = {
    '=' : 1,
    '<' : 2,
    '>' : 2,
    '+' : 3,
    '-' : 3,
    '*' : 4,
    '/' : 4
}

class AST():
    pass

class Statement(AST):
    def __init__(self, expressions: List[AST]):
        self.expressions = expressions

class Number(AST):
    def __init__(self, value: Token):
        self.value = value
    def __repr__(self):
        return 'Number(' + str(self.value.value) + ')'

class BinaryOperator(AST):
    def __init__(self, left: AST, operator: Token, right: AST):
        self.left = left
        self.operator = operator
        self.right = right
    def __repr__(self):
        return 'BinaryOperator{ ' + str(self.left) + ' ' + str(self.operator.value) + ' ' + str(self.right) + ' }'

def parseBinaryOperator(lhs: AST, tokenList: List[Token]) -> Union[BinaryOperator, Error, Tuple[BinaryOperator, BinaryOperator]]:
    # tokenlist is tokens after the +
    head, *tail = tokenList
    if tail[0].type != 'NUMBER' and tail[0].type != 'IDENTIFIER' and tail[0].type != 'BLOCK':
        return Error('Invalid syntax', head.line, head.position)

    elif tail[1].type == 'END':
        if len(tail[2:]):
             return BinaryOperator(lhs, head, Number(tail[0])), tail[2:]
        else:
            return BinaryOperator(lhs, head, Number(tail[0])), tail[1:]

    elif tail[1].type == 'ADD' or tail[1].type == 'SUBTRACT' or tail[1].type == 'MULTIPLY' or tail[1].type == 'DIVIDE':
        if PRECEDENCE[head.value] > PRECEDENCE[tail[1].value]:
            test = BinaryOperator(lhs, head, Number(tail[0]))
            nextBinaryOperator = parseBinaryOperator(test, tail[1:])
            return nextBinaryOperator
        else:
            nextBinaryOperator = parseBinaryOperator(Number(tail[0]), tail[1:])
            return BinaryOperator(lhs, head, nextBinaryOperator[0]), nextBinaryOperator[1]

    else:
        return Error('Invalid syntax', head.line, head.position)

def parseGeneral(tokenList: List[Token], prev: List[AST] = []) -> Union[Error, BinaryOperator]:
    if len(tokenList) > 0:
        head, *tail = tokenList

        if head.type == 'NUMBER' and prev == []:
            prev.append(Number(head))
            result = parseGeneral(tail, prev)
        elif head.type == 'NUMBER' and prev != []:
            result = Error('Syntax error', prev[0].value.line, prev[0].value.position)

        elif head.type == 'ADD' or head.type == 'SUBTRACT' or head.type == 'MULTIPLY' or head.type == 'DIVIDE':
            lhs = prev.pop()
            expression = parseBinaryOperator(lhs, tokenList)
            list: List[Token] = expression[1]
            result = expression[0], parseGeneral(list)
        else:
            result = 'EOF' # TODO

    else:
        result = None # TODO

    return result

def parse(tokenList: List[Token], prev = None):
    result =  parseGeneral(tokenList)
    return result

# ---------------------------------------------
# Run/Debug
# ---------------------------------------------
# for i in (lex(sourceCode)):
#     print(i)

print(parse(lex(sourceCode)))
# parse(lex(sourceCode))
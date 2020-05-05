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

# This function returns a the token-type of a certain string
def getTokenType(value: str) -> str:
    tokenTypes = {
        'ADD'               : r'[+]',
        'MULTIPLY'          : r'[*]',
        'DIVIDE'            : r'[/]',
        'SUBTRACT'          : r'[-]',
        'ASSIGN'           : r'[=]',
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
# This function creates a token (using getTokenType) and returns a function that adds that token to a list.
# createToken :: str -> Tuple[int,int] -> Callable[List[Token]] -> Union[List[Token], None]
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

# This function returns a function to update to either a newline, or a next character.
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

# This function converts the sourcecode to tokens.
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

class Identifier(AST):
    def __init__(self, value: Token):
        self.value = value
    def __repr__(self):
        return 'Identifier(' + str(self.value.value) + ')'

class Assign(AST):
    def __init__(self, variable: Identifier, value: AST):
        self.variable = variable
        self.value = value
    def __repr__(self):
        return 'Assign{' + str(self.variable) + '=' + str(self.value) + '}'

class BinaryOperator(AST):
    def __init__(self, left: AST, operator: Token, right: AST):
        self.left = left
        self.operator = operator
        self.right = right
    def __repr__(self):
        return 'BinaryOperator{ ' + str(self.left) + ' ' + str(self.operator.value) + ' ' + str(self.right) + ' }'

# ---------------------------------------------
# Parse functions
# ---------------------------------------------
def parseBinaryOperator(lhs: AST, tokenList: List[Token]) -> Union[BinaryOperator, Error, Tuple[BinaryOperator, BinaryOperator]]:
    head, *tail = tokenList

    if tail[0].type != 'NUMBER' and tail[0].type != 'IDENTIFIER' and tail[0].type != 'BLOCK':
        return Error('Invalid syntax', head.line, head.position)

    elif tail[1].type == 'END':
        if type(lhs) == BinaryOperator and PRECEDENCE[head.value] > PRECEDENCE[lhs.operator.value]:
            newBinaryOperator = BinaryOperator(lhs.left, lhs.operator, BinaryOperator(lhs.right, head, Number(tail[0])))
            if len(tail[2:]):
                return newBinaryOperator, tail[2:]
            else:
                return newBinaryOperator, tail[1:]
        else:
            newBinaryOperator = BinaryOperator(lhs, head, Number(tail[0]))
            return newBinaryOperator, tail[1:]

    elif tail[1].type == 'ADD' or tail[1].type == 'SUBTRACT' or tail[1].type == 'MULTIPLY' or tail[1].type == 'DIVIDE':
        if type(lhs) == BinaryOperator and PRECEDENCE[head.value] > PRECEDENCE[lhs.operator.value]:
            nextBinaryOperator = parseBinaryOperator(lhs.right, tokenList)
            newBinaryOperator = BinaryOperator(lhs.left, lhs.operator, nextBinaryOperator[0])
            return newBinaryOperator, nextBinaryOperator[1]
        else:
            newBinaryOperator = BinaryOperator(lhs, head, Number(tail[0]))
            nextBinaryOperator = parseBinaryOperator(newBinaryOperator, tail[1:])
            return nextBinaryOperator

    else:
        return Error('Invalid syntax', head.line, head.position)


def parseAssign(lhs: AST, tokenlist: List[Token]) -> Tuple[Assign, List[Token]]:
    variable: Identifier = lhs
    rhs: Union[List[Union[AST, List[Token]]], Error] = parseGeneral(tokenlist)
    result: Assign = Assign(variable, rhs[0])
    return result, rhs[1]

def parseGeneral(tokenList: List[Token], prev: List[AST] = []) -> Tuple[Tuple[AST], List[Token]]:
    if len(tokenList) > 0:
        head, *tail = tokenList
        if head.type == 'NUMBER' and not len(prev):
            prev.append(Number(head))
            result = parseGeneral(tail, prev)
        elif head.type == 'NUMBER' and prev != []:
            result = Error('Syntax error', prev[0].value.line, prev[0].value.position)

        elif head.type == 'IDENTIFIER':
            if not len(prev):
                prev.append(Identifier(head))
                expression = parseGeneral(tail, prev)
                result = expression
            else:
                result = Error('expected operation or assingment', head.line, head.position)

        elif head.type == 'ADD' or head.type == 'SUBTRACT' or head.type == 'MULTIPLY' or head.type == 'DIVIDE':
            if len(prev):
                lhs = prev.pop()
                expression = parseBinaryOperator(lhs, tokenList)
                result = expression[0], expression[1]
            else:
                result = Error('Expected left hand side of operator', head.line, head.position)

        elif head.type == 'ASSIGN':
            if len(prev):
                variable = prev.pop()
                expression = parseAssign(variable, tail)
                result = expression[0], expression[1]

        elif head.type == 'END':
            result = parseGeneral(tail)

        else:
            result = 'EOF' # TODO

    else:
        result = 'EOF' # TODO

    return result

def parse(tokenList: List[Token]):
    result: Tuple[Tuple[AST], List[Token]] =  parseGeneral(tokenList)
    if len(result) > 1 and len(result[1]) > 1:
        nextResult = parse(result[1])
        return (result[0],) + nextResult
    else:
        return result[0], 'EOF'
# ---------------------------------------------
# Run/Debug
# ---------------------------------------------
# for i in (lex(sourceCode)):
#     print(i)

print(parse(lex(sourceCode)))
# parse(lex(sourceCode))
from typing import List, Tuple, Callable, Union
from copy import deepcopy
import re
import operator

# Reading input file
inputFile = open("input.txt", "r")
sourceCode= inputFile.read()

# ---------------------------------------------
# Classes
# ---------------------------------------------
class AST():
    pass

class Error(AST):
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
        'COMPARE'           : r'[<>]',
        'ASSIGN'            : r'[=]',
        'BLOCK'             : r'[()]',
        'END'               : r';',
        'IF'                : r'^aint$',
        'TRUE'              : r'^yea$',
        'FALSE'             : r'^nah$',
        'WHILE'             : r'^whilst$',
        'EXECUTE'           : r'^execute$',
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

class Block(AST):
    def __init__(self, expressions: List[AST]):
        self.expressions = expressions
    def __repr__(self):
        return 'Block[' + str(self.expressions) + ']'

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

class WhileStatement(AST):
    def __init__(self, condition: Block, ifTrue: Block):
        self.condition = condition
        self.ifTrue = ifTrue
    def __repr__(self):
        return 'While[' + str(self.condition) + ': ifTrue(' + str(self.ifTrue) + ')'

class IfStatement(AST):
    def __init__(self, condition: Block, ifTrue: Block = None, ifFalse: Block = None):
        self.condition = condition
        self.ifTrue = ifTrue
        self.ifFalse = ifFalse
    def __repr__(self):
        return 'if[' + str(self.condition) + ': ifTrue(' + str(self.ifTrue) + '), ifFalse(' + str(self.ifFalse) + ')]'

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

# This function parses binary operators. where head is the operator
def parseBinaryOperator(lhs: AST, tokenList: List[Token]) -> Union[BinaryOperator, Error, Tuple[BinaryOperator, List[Token]]]:
    head, *tail = tokenList
    # return Error if following character is not a block, number or identifier
    if type(tail[0]) != Block and tail[0].type != 'NUMBER' and tail[0].type != 'IDENTIFIER' and tail[0].type != 'BLOCK':
        return Error('Expected a Number value', head.line, head.position)

    # if this is the last operator return it with the tail, else return a binary operator with the next binary operator
    elif tail[1].type == 'END':
        if type(lhs) == BinaryOperator and PRECEDENCE[head.value] > PRECEDENCE[lhs.operator.value]:
            nextParsedToken = parseExpression(tail[0:2])[0][0]
            if type(nextParsedToken) == Error:
                nextParsedToken: Error
                return nextParsedToken

            newBinaryOperator = BinaryOperator(lhs.left, lhs.operator, BinaryOperator(lhs.right, head, nextParsedToken))
            if len(tail[2:]):
                return newBinaryOperator, tail[2:]
            else:
                return newBinaryOperator, tail[1:]
        else:
            parsedRhs = parseExpression(tail[0:2])[0][0]
            if type(parsedRhs) == Error:
                parsedRhs: Error
                return parsedRhs
            else:
                newBinaryOperator = BinaryOperator(lhs, head, parsedRhs)
                return newBinaryOperator, tail[1:]

    # if the next operator has a higher precedence return a binary operator of lhs(.left) and the next binary operator
    # else return the next binary operator with the current binary operator as lhs
    elif tail[1].type == 'ADD' or tail[1].type == 'SUBTRACT' or tail[1].type == 'MULTIPLY' or tail[1].type == 'DIVIDE' or tail[1].type == 'COMPARE':
        if type(lhs) == BinaryOperator and PRECEDENCE[head.value] > PRECEDENCE[lhs.operator.value]:
            lhs: BinaryOperator
            nextBinaryOperator = parseBinaryOperator(lhs.right, tokenList)
            if type(nextBinaryOperator) == Error:
                nextBinaryOperator: Error
                return nextBinaryOperator
            else:
                nextBinaryOperator[0]: BinaryOperator
                if PRECEDENCE[nextBinaryOperator[0].operator.value] > PRECEDENCE[lhs.operator.value]:
                    newBinaryOperator = BinaryOperator(lhs.left, lhs.operator, nextBinaryOperator[0])
                    return newBinaryOperator, nextBinaryOperator[1]
                else:
                    newBinaryOperator = BinaryOperator(BinaryOperator(lhs.left, lhs.operator, nextBinaryOperator[0].left), nextBinaryOperator[0].operator, nextBinaryOperator[0].right)
                    return newBinaryOperator, nextBinaryOperator[1]
        else:
            parsedRhs = parseExpression([tail[0], Token('END', ';', head.line, head.position + 1)])[0][0]
            if type(parsedRhs) == Error:
                parsedRhs: Error
                return parsedRhs
            else:
                parsedRhs: AST
                newBinaryOperator = BinaryOperator(lhs, head, parsedRhs) # create BinOp of lhs, head and parsed next token.
                nextBinaryOperator = parseBinaryOperator(newBinaryOperator, tail[1:])
                return nextBinaryOperator

    # if the next token is a block, parse the block and return a binary operator with that block as rhs
    elif tail[0].type == 'BLOCK':
        nextBlock = parseBlock(tail)
        if len(nextBlock[1]) > 1:
            return parseBinaryOperator(BinaryOperator(lhs, head, nextBlock[0]), nextBlock[1])
        else:
            return BinaryOperator(lhs, head, nextBlock[0]), []

    else:
        return Error('Invalid syntax', head.line, head.position)


def parseAssign(lhs: AST, tokenlist: List[Token]) -> Union[Error, Tuple[Assign, List[Token]] ]:
    variable: Identifier = lhs
    rhs: Union[List[Union[AST, List[Token]]], Error] = parseExpression(tokenlist)
    if type(rhs) == Error:
        rhs: Error
        return rhs
    elif type(rhs[0]) == list:
        result: Assign = Assign(variable, rhs[0][0])
        return result, rhs[1]
    else:
        result: Assign = Assign(variable, rhs[0])
        return result, rhs[1]

def parseBlock(tokenList: List[Token], prev: List[Token] = []) -> Union[Error, Tuple[Block, List[Token]] ]:
    head, *tail = tokenList
    tokens = prev.copy()

    if type(head) == Block:
        tokens.append(head)
        result = parseBlock(tail, tokens)

    elif head.value == '(':
        if len(tokens) == 0:
            result = parseBlock(tail, [head])
        else:
            parsedBlock = parseBlock(tail, [head])
            if type(parsedBlock) == Error:
                return parsedBlock
            else:
                tokens.append(parsedBlock[0])
                tokens += parsedBlock[1]
                result = parseExpression(tokens)
    elif head.value == ')':
        if tokens[0].value == '(':
            tokens.append(Token('END', ';', head.line, head.position))
            expressions = parse(tokens[1:])
            if type(expressions) == Error:
                return expressions
            else:
                result = Block(expressions[:-1]), tail
        else:
            print('ohooh') # TODO
    elif head == 'EOF':
        result = Error('Expected \')\'', head.line, head.position)
    else:
        tokens.append(head)
        result = parseBlock(tail, tokens)

    return result

def parseWhile(tokenList: List[Token]):
    head, *tail = tokenList

    if head.type == 'WHILE' and tail[0].type == 'BLOCK':
        condition, tailAfterBlock = parseBlock(tail)
        if tailAfterBlock[0].type == 'EXECUTE': # if first token in tail of parsed block is a EXECUTE token
            ifTrue, tailAfterExecute = parseWhile(tailAfterBlock)
            result = WhileStatement(condition, ifTrue), tailAfterExecute
        else:
            result = Error('Exprected execute statement after while', head.line, head.position)

    elif head.type == 'EXECUTE' and tail[0].type == 'BLOCK':
        result = parseBlock(tail)

    else:
        result = Error('Expected \'(\'', tail[0].line, tail[0].position)

    return result

def parseIf(tokenList: List[Token]):
    head, *tail = tokenList

    if head.type == 'IF' and tail[0].type == 'BLOCK':
        parsedBlock = parseBlock(tail)
        condition = parsedBlock[0]

        parsedFirstStatement = parseIf(parsedBlock[1])
        if parsedFirstStatement == None:
            return Error('Expected yea or nah block after if statement', head.line, head.position)

        parsedFirstStatement: Tuple[IfStatement, List[Token]]
        parsedSecondStatement = parseIf(parsedFirstStatement[1])

        if parsedSecondStatement == None:
            result = IfStatement(condition, parsedFirstStatement[0].ifTrue, parsedFirstStatement[0].ifFalse), parsedFirstStatement[1]
        elif parsedFirstStatement[0].ifTrue == None and parsedSecondStatement[0].ifFalse == None:   # if second was ifTrue and first was ifFalse
            result = IfStatement(condition, parsedSecondStatement[0].ifTrue, parsedFirstStatement[0].ifFalse), parsedSecondStatement[1]
        elif parsedSecondStatement[0].ifTrue == None and parsedFirstStatement[0].ifFalse == None:   # if first was ifTrue and second was ifFalse
            result = IfStatement(condition, parsedFirstStatement[0].ifTrue, parsedSecondStatement[0].ifFalse), parsedSecondStatement[1]
        else:
            result = Error('Something went wrong with if statement', head.line, head.position)


    elif head.type == 'IF' and tail[0].type != 'BLOCK':
        result = Error('if statement needs to be followed by a code block', head.line, head.position)

    elif head.type == 'TRUE':   # TODO Check if next = false
        parsedBlock = parseBlock(tail)
        ifTrue = parsedBlock[0]
        result = IfStatement(Block([]), ifTrue), parsedBlock[1]

    elif head.type == 'FALSE':  # TODO check if next = true
        parsedBlock = parseBlock(tail)
        ifFalse = parsedBlock[0]
        result = IfStatement(Block([]), ifFalse=ifFalse), parsedBlock[1]

    else:
        result = None



    return result



def parseExpression(tokenList: List[Token], last: List[AST] = []) -> Union[Error, Tuple[Tuple[AST], List[Token]]]:
    prev = last.copy()
    if len(tokenList) > 0:
        head, *tail = tokenList

        if type(head) == Block:
            result = head, tail

        elif head.type == 'NUMBER' and not len(prev):
            prev.append(Number(head))
            result = parseExpression(tail, prev)
        elif head.type == 'NUMBER' and prev != []:
            result = Error('Syntax error', prev[0].value.line, prev[0].value.position) # TODO (wat voor error?)

        elif head.type == 'IDENTIFIER':
            if not len(prev):
                prev.append(Identifier(head))
                expression = parseExpression(tail, prev)
                result = expression
            else:
                result = Error('expected operation or assingment', head.line, head.position)

        elif head.type == 'ADD' or head.type == 'SUBTRACT' or head.type == 'MULTIPLY' or head.type == 'DIVIDE' or head.type == 'COMPARE':
            if len(prev):
                lhs = prev.pop()
                expression = parseBinaryOperator(lhs, tokenList)
                result = expression
            else:
                result = Error('Expected left hand side of operator', head.line, head.position)

        elif head.type == 'ASSIGN':
            if len(prev):
                variable = prev.pop()
                expression = parseAssign(variable, tail)
                result = expression

        elif head.type == 'IF':
            result = parseIf(tokenList)

        elif head.type == 'TRUE' or head.type == 'FALSE':
            result = Error('Expected if-statement', head.line, head.position)

        elif head.type == 'WHILE':
            result = parseWhile(tokenList)

        elif head.type == 'EXECUTE':
            result = Error('Expected while-statement', head.line, head.position)

        elif head.type == 'BLOCK':
            result = parseBlock(tokenList)

        elif head.type == 'END':
            if len(prev):
                result = prev, tail

            else:
                result = parseExpression(tail)

        else:
            result = 'EOF' # TODO

    else:
        result = 'EOF' # TODO

    return result

def parse(tokenList: List[Token]):
    result: Tuple[Tuple[AST], List[Token]] = parseExpression(tokenList)
    if type(result) == Error:
        return result
    elif len(result) > 1 and len(result[1]) > 1:
        nextResult = parse(result[1])
        if type(nextResult) == Error:
            return nextResult
        else:
            return (result[0],) + nextResult
    else:
        return result[0], 'EOF'


# ---------------------------------------------
# Interpreter
# ---------------------------------------------
class State:
    def __init__(self):
        self.variables = {}
        self.errors = []
    def __repr__(self):
        return 'State(Variables: ' + str(self.variables) + ', errors: ' + str(self.errors) +  ')'

def visitNumber(node: Number, originalState: State) -> Tuple[int, State]:
    return int(node.value.value), originalState

def visitIdentifier(node: Identifier, originalState: State) -> Union[Tuple[None, State], Tuple[int, State]]:
    newState = originalState
    variableValue = originalState.variables.get(node.value.value)
    if variableValue == None:
        newState.errors.append(Error('Expected value', node.value.line, node.value.position))
        return None, newState
    else:
        return variableValue, newState


def visitAssign(node: Assign, originalState: State) -> State:
    value = visit(node.value, originalState)
    if value[0] == None:
        newState = value[1]
        return newState
    else:
        newState = originalState
        newState.variables.update({node.variable.value.value : value[0]})
        return newState


def visitBinaryOperator(node: BinaryOperator, originalState: State) -> Tuple[int, State]:
    newState = deepcopy(originalState)
    lhs = visit(node.left, originalState)[0]    # TODO if visit Number changes state, this needs work
    rhs = visit(node.right, originalState)[0]
    operators = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "<": operator.lt,
        ">": operator.gt
    }
    operatorFunction = operators[node.operator.value]
    if (type(lhs) == int or type(lhs) == float) and (type(rhs) == int or type(rhs) == float):
        return operatorFunction(lhs, rhs), newState
    else:
        return Error('Expected Number before operator' , node.operator.line, node.operator.position)    #TODO

def visitWhileStatement(node: WhileStatement, originalState: State):
    condition = visit(node.condition, originalState)
    if type(condition) == Error:
        newState = deepcopy(originalState)
        newState.errors.append(condition)
        result = newState

    elif condition[0] == True:
        result = visit(node.ifTrue, originalState)
        if type(result) == State:
            result = visitWhileStatement(node, result)
        else:
            nextLoop = visitWhileStatement(node, result[-1])
            if type(nextLoop) == State:
                result = result[:-1] + (nextLoop,)
            else:
                result = result[:-1] + nextLoop
    else:
        result = originalState

    return result

def visitIfStatement(node: IfStatement, originalState: State):
    condition = visit(node.condition, originalState)
    if condition[0] == True:
        result = visit(node.ifTrue, originalState)
    else:
        result = visit(node.ifFalse, originalState)
    return result

def visitBlock(node: Block, originalState: State):
    # voor elk item in een block. visit dat item
    head, *tail = node.expressions
    firstNode = visit(head, originalState)
    if len(tail) > 0:
        if type(firstNode) == State:
            nextParsedBlock = visit(Block(tail), firstNode)
            return nextParsedBlock
        else:
            return (firstNode[0],) + visit(Block(tail), firstNode[1])

    return firstNode

def visit(node: AST, originalState: State):
    if type(node) == BinaryOperator:
        node: BinaryOperator
        return visitBinaryOperator(node, originalState)
    elif type(node) == Number:
        node: Number
        return visitNumber(node, originalState)
    elif type(node) == Assign:
        node: Assign
        return visitAssign(node, originalState)
    elif type(node) == Identifier:
        node: Identifier
        return visitIdentifier(node, originalState)
    elif type(node) == WhileStatement:
        node: WhileStatement
        return visitWhileStatement(node, originalState)
    elif type(node) == IfStatement:
        node: IfStatement
        return visitIfStatement(node, originalState)
    elif type(node) == Block:
        node: Block
        return visitBlock(node, originalState)
    else:
        print('dit gaat fout(node niet bekend: ' + str(node))
        return node, originalState # TODO check for correct behaviour


def interpret(ast: List[AST], originalState: State) -> Tuple[Union[int, State], State]:
    newState = deepcopy(originalState)
    head, *tail = ast

    if tail != 'EOF' and len(tail) > 1:
        currentExpression = visit(head, newState)
        if type(currentExpression) == State:
            return interpret(tail, currentExpression)
        else:
            nextExpression = interpret(tail, currentExpression[-1])
            if type(nextExpression) == State:                               # TODO miss niet nodig, else kan ook als erboven geschreven worden
                return currentExpression[:-1] + (nextExpression,)
            else:
                return currentExpression[:-1] + nextExpression
    elif tail[0] == 'EOF':
        return visit(head, newState)
    else:
        pass # has to return an error



# ---------------------------------------------
# Run/Debug
# ---------------------------------------------
# for i in (lex(sourceCode)):
#     print(i)

originalState = State()
# parse(lex(sourceCode))
# print(parse(lex(sourceCode)))
print(interpret(parse(lex(sourceCode)), originalState))

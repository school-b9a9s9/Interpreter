from typing import List, Tuple, Callable, Union
from copy import deepcopy
import re
import operator
import sys

# --------------------------------------------
# Decorator
# --------------------------------------------
def debug(function: Callable[[any], any]) -> Callable[[any], any]:
    def inner(*args, **kwargs):
        inner.called += 1
        inner.arguments.append(args)

        returnedValue = function(*args, **kwargs)

        inner.returnValues.append(returnedValue)

        return returnedValue

    inner.called = 0
    inner.arguments = []
    inner.returnValues = []
    inner.getCallCount = lambda : str(function.__name__) + ' is called ' + str(inner.called) + ' times'
    inner.getArguments = lambda : str(function.__name__) + ' is called with arguments: ' + str(inner.arguments)
    inner.getReturnValues = lambda : str(function.__name__) + ' returned ' + str(inner.returnValues)
    inner.getStats = lambda : inner.getCallCount() + '\n' + inner.getArguments() + '\n' + inner.getReturnValues() + '\n'
    return inner

# ---------------------------------------------
# Classes
# ---------------------------------------------
# Superclass for all AST nodes
class AST():
    pass

# Classes to represent Errors
class Error(AST):
    def __init__(self, errorMessage: str, line: int, position: int):
        self.errorMessage = errorMessage
        self.line = line
        self.position = position

    def __repr__(self):
        return self.errorMessage + ' at line: ' + str(self.line) + ', position: ' + str(self.position)

class GlobalError(AST):
    def __init__(self, errorMessage: str):
        self.errorMessage = errorMessage
    def __repr__(self):
        return self.errorMessage

# Class to represent tokens
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

# This function returns the token-type of a certain string
# getTokenType :: str -> str
def getTokenType(value: str) -> str:
    tokenTypes = {
        'ADD'               : r'[+]',
        'MULTIPLY'          : r'[*]',
        'DIVIDE'            : r'[/]',
        'SUBTRACT'          : r'[-]',
        'COMPARE'           : r'[<>]|^eq$',
        'ASSIGN'            : r'[=]',
        'BLOCK'             : r'[()]',
        'END'               : r';',
        'IF'                : r'^aint$',
        'TRUE'              : r'^yea$',
        'FALSE'             : r'^nah$',
        'WHILE'             : r'^whilst$',
        'EXECUTE'           : r'^execute$',
        'PRINT'             : r'^display$',
        'NUMBER'            : r'^[0-9]*$',
        'IDENTIFIER'        : r'\w'             # Has to be last
    }
    tokenTypeMatch = list(filter(lambda tokenType: re.match(tokenTypes.get(tokenType), value), tokenTypes))
    if len(tokenTypeMatch) > 0:
        return tokenTypeMatch[0]
    else:
        return "UNKNOWN"


# This function creates a token (using getTokenType) and returns a function that adds that token to a list.
# If no token is created, this function returns the original tokenList
# createToken :: str -> Tuple[int,int] -> Union[Callable, Callable[[List[Token], List[Token]]]]
def createToken(value: str, position: Tuple[int, int]) -> Callable[[List[Token]], List[Token]]:
    def addTokenToList(tokenList: List[Token]) -> List[Token]:
        newTokenList = deepcopy(tokenList)
        newTokenList.append(token)
        return newTokenList
    def doNothing(tokenList: List[Token]) -> List[Token]:
        return tokenList

    if len(value) == 0:
        return doNothing
    elif not str.isdigit(value) and not str.isalpha(value):
        token = Token(getTokenType(value), value, position[0], position[1])
    else:
        startPosition = (position[0], position[1] - len(value))
        token = Token(getTokenType(value), value, startPosition[0], startPosition[1])
    return addTokenToList

# This function returns a function to update to either a newline, or a next character.
# updateCurrentPosition :: str -> Callable[[Tuple[int, int]], Tuple[int, int]]
def updateCurrentPosition(currentCharacter: str) -> Callable[[Tuple[int, int]], Tuple[int, int]]:
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
# lex :: str -> str -> Tuple[int, int] -> List[Token]
def lex(sourceCode: str, prev: str = '', currentPosition: Tuple[int, int] = (1, 1)) -> List[Token]: # sourceCode: code, s: not-tokens to be converted a token
    head, *tail = sourceCode
    tokens: List(Token) = []
    lastCharacters = deepcopy(prev)

    if str.isalpha(head) or str.isdigit(head):
        lastCharacters += head
    else:
        if len(lastCharacters) > 0:
            tokens = createToken(lastCharacters, currentPosition)(tokens)
        lastCharacters = ''

        if head != ' ' and head != '\n':
            tokens = createToken(head, currentPosition)(tokens)

    nextPosition = updateCurrentPosition(head)(currentPosition)

    if len(tail) > 0:
        if tokens:
            return tokens + lex(tail, lastCharacters, nextPosition)
        else:
            return lex(tail, lastCharacters, nextPosition)
    else:
        createToken(lastCharacters, currentPosition)(tokens)
        return tokens


# ---------------------------------------------
# Parser
# ---------------------------------------------
PRECEDENCE = {
    'eq' : 1,
    '<' : 2,
    '>' : 2,
    '+' : 3,
    '-' : 3,
    '*' : 4,
    '/' : 4
}

# AST node classes:

# Class to represent a code block with one or more expressions
class Block(AST):
    def __init__(self, expressions: List[AST]):
        self.expressions = expressions
    def __repr__(self):
        return 'Block[' + str(self.expressions) + ']'

# Class to represent a number value
class Number(AST):
    def __init__(self, value: Token):
        self.value = value
    def __repr__(self):
        return 'Number(' + str(self.value.value) + ')'

# Class to represent an identifier (variables)
class Identifier(AST):
    def __init__(self, value: Token):
        self.value = value
    def __repr__(self):
        return 'Identifier(' + str(self.value.value) + ')'

# Class to represent an assignment node
class Assign(AST):
    def __init__(self, variable: Identifier, value: AST):
        self.variable = variable
        self.value = value
    def __repr__(self):
        return 'Assign{' + str(self.variable) + '=' + str(self.value) + '}'

# Class to represent a while statement
class WhileStatement(AST):
    def __init__(self, condition: Block, ifTrue: Block):
        self.condition = condition
        self.ifTrue = ifTrue
    def __repr__(self):
        return 'While[' + str(self.condition) + ': ifTrue(' + str(self.ifTrue) + ')'

# Class to represent an if statement
class IfStatement(AST):
    def __init__(self, condition: Block, ifTrue: Block = None, ifFalse: Block = None):
        self.condition = condition
        self.ifTrue = ifTrue
        self.ifFalse = ifFalse
    def __repr__(self):
        return 'if[' + str(self.condition) + ': ifTrue(' + str(self.ifTrue) + '), ifFalse(' + str(self.ifFalse) + ')]'

# Class to represent a binary operator
class BinaryOperator(AST):
    def __init__(self, left: AST, operator: Token, right: AST):
        self.left = left
        self.operator = operator
        self.right = right
    def __repr__(self):
        return 'BinaryOperator{ ' + str(self.left) + ' ' + str(self.operator.value) + ' ' + str(self.right) + ' }'

# Class to represent a print statement
class PrintStatement(AST):
    def __init__(self, result: AST):
        self.result = result
    def __repr__(self):
        return 'Print[' + str(self.result) + ']'


# ---------------------------------------------
# Parse functions
# ---------------------------------------------

# This function parses binary operators, where head is the operator
# parseBinaryOperator :: AST -> List[Tokens] -> Union[BinaryOperator, Error, Tuple[BinaryOperator, List[Token]]]
def parseBinaryOperator(lhs: AST, tokenList: List[Token]) -> Union[BinaryOperator, Error, Tuple[BinaryOperator, List[Token]]]:
    head, *tail = tokenList
    # return Error if following character is not a block, number or identifier
    if type(tail[0]) != Block and tail[0].type != 'NUMBER' and tail[0].type != 'IDENTIFIER' and tail[0].type != 'BLOCK':
        return Error('Expected a Number value', head.line, head.position)

    # if this is the last operator return it with the tail, else return a binary operator with the next binary operator
    elif tail[1].type == 'END':
        # if current precedence is higher than the precedence of the lhs (example: 3+3*9)
        if type(lhs) == BinaryOperator and PRECEDENCE[head.value] > PRECEDENCE[lhs.operator.value]:
            lhs: BinaryOperator
            nextParsedToken = parseExpression(tail[0:2])[0][0]
            if type(nextParsedToken) == Error:
                nextParsedToken: Error
                return nextParsedToken
            # with example 3+3*9, returns binop(3 + binop(3 * 9))
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
                # with example 3*3+9, returns binop(binop(3*3) + 9)
                newBinaryOperator = BinaryOperator(lhs, head, parsedRhs)
                return newBinaryOperator, tail[1:]

    # if there is an operator after head.
    elif tail[1].type == 'ADD' or tail[1].type == 'SUBTRACT' or tail[1].type == 'MULTIPLY' or tail[1].type == 'DIVIDE' or tail[1].type == 'COMPARE':
        # if the next operator has a higher precedence return a binary operator of lhs(.left) and the next binary operator
        if type(lhs) == BinaryOperator and PRECEDENCE[head.value] > PRECEDENCE[lhs.operator.value]:
            lhs: BinaryOperator
            nextBinaryOperator = parseBinaryOperator(lhs.right, tokenList)
            if type(nextBinaryOperator) == Error:
                nextBinaryOperator: Error
                return nextBinaryOperator
            else:
                nextBinaryOperator[0]: BinaryOperator
                if PRECEDENCE[nextBinaryOperator[0].operator.value] > PRECEDENCE[lhs.operator.value]:
                    # with example (3+3*9), return binop(3 + binop(3*9))
                    newBinaryOperator = BinaryOperator(lhs.left, lhs.operator, nextBinaryOperator[0])
                    return newBinaryOperator, nextBinaryOperator[1]
                else:
                    # with example (3*3+9), return binop(binop(3*3) + 9)
                    newBinaryOperator = BinaryOperator(BinaryOperator(lhs.left, lhs.operator, nextBinaryOperator[0].left), nextBinaryOperator[0].operator, nextBinaryOperator[0].right)
                    return newBinaryOperator, nextBinaryOperator[1]
        else:
            parsedRhs = parseExpression([tail[0], Token('END', ';', head.line, head.position + 1)])[0]
            if type(parsedRhs) != Block:
                parsedRhs = parsedRhs[0]
            if type(parsedRhs) == Error:
                parsedRhs: Error
                return parsedRhs
            else:
                # with example (3+3+9), return binop(3 + parseExpression(3+9)
                parsedRhs: AST
                newBinaryOperator = BinaryOperator(lhs, head, parsedRhs) # create BinOp of lhs, head and parsed next token.
                nextBinaryOperator = parseBinaryOperator(newBinaryOperator, tail[1:])
                return nextBinaryOperator

    # if the next token is a block, parse the block and return a binary operator with that block as rhs
    elif tail[0].type == 'BLOCK':
        nextBlock = parseBlock(tail)
        if len(nextBlock[1]) > 1 and (nextBlock[1][0].type == 'ADD' or nextBlock[1][0].type == 'SUBTRACT' or nextBlock[1][0].type == 'MULTIPLY' or nextBlock[1][0].type == 'DIVIDE' or nextBlock[1][0].type == 'COMPARE'):
            return parseBinaryOperator(BinaryOperator(lhs, head, nextBlock[0]), nextBlock[1])
        else:
            return BinaryOperator(lhs, head, nextBlock[0]), nextBlock[1]

    else:
        return Error('Invalid syntax', head.line, head.position)

# parseAssign :: AST -> List[Token] -> Unionp[Error, Tuple[Assign, List[Token]]]
def parseAssign(lhs: AST, tokenList: List[Token]) -> Union[Error, Tuple[Assign, List[Token]] ]:
    variable: Identifier = lhs
    rhs = parseExpression(tokenList)
    if type(rhs) == Error:
        rhs: Error
        return rhs
    elif type(rhs[0]) == list:
        result: Assign = Assign(variable, rhs[0][0])
        return result, rhs[1]
    else:
        result: Assign = Assign(variable, rhs[0])
        return result, rhs[1]

# parseBlock :: List[Token] -> List[Token] -> Union[Error, Tuple[Block, List[Token]]]
def parseBlock(tokenList: List[Token], prev: List[Token] = None) -> Union[Error, Tuple[Block, List[Token]] ]:
    if prev == None:
        prev = []

    head, *tail = tokenList
    tokens = deepcopy(prev)

    # a block inside a block is added to the token list
    if type(head) == Block:
        tokens.append(head)
        result = parseBlock(tail, tokens)

    # on the first '(', parsing a block is started
    elif head.value == '(':
        if len(tokens) == 0:
            result = parseBlock(tail, [head])
        # if this is a block inside a block, parse the subblock and add it to the token list
        else:
            parsedBlock = parseBlock(tail, [head])
            if type(parsedBlock) == Error:
                return parsedBlock
            else:
                tokens.append(parsedBlock[0])
                tokens += parsedBlock[1]
                result = parseExpression(tokens)

    # the end of a block
    elif head.value == ')':
        if len(tokens) > 0 and tokens[0].value == '(':
            if type(tokens[-1]) == Token and tokens[-1].type != 'END':    # Add END token if it isn't there
                tokens.append(Token('END', ';', head.line, head.position))

            expressions = parse(tokens[1:])
            if type(expressions) == Error:
                return expressions
            else:
                result = Block(expressions[:-1]), tail
        else:
            result = Error('expected \'(\' before \')\'', head.line, head.position)
    elif head == 'EOF':
        result = Error('Expected \')\'', head.line, head.position)
    else:
        tokens.append(head)
        result = parseBlock(tail, tokens)

    return result

# parseWhile :: List[Token] -> Union[Error, Tuple[WhileStatement, List[Token]], Block]
def parseWhile(tokenList: List[Token]) -> Union[Error, Tuple[WhileStatement, List[Token]], Block]:
    head, *tail = tokenList

    # if head is a while token and is followed by a code block
    if head.type == 'WHILE' and tail[0].type == 'BLOCK':
        condition, tailAfterBlock = parseBlock(tail)
        if tailAfterBlock[0].type == 'EXECUTE': # if first token in tail of parsed block is an EXECUTE token
            ifTrue, tailAfterExecute = parseWhile(tailAfterBlock)
            result = WhileStatement(condition, ifTrue), tailAfterExecute
        else:
            result = Error('Expected execute statement after while', head.line, head.position)

    # if head is an execute token and followed by a code block
    elif head.type == 'EXECUTE' and tail[0].type == 'BLOCK':
        result = parseBlock(tail)

    else:
        result = Error('Expected \'(\'', tail[0].line, tail[0].position)

    return result

def parseIf(tokenList: List[Token]) -> Union[Error, Tuple[IfStatement, List[Token]], IfStatement]:
    head, *tail = tokenList

    # if block is already parsed, for example when if inside while
    if head.type == 'IF' and type(tail[0]) == Block:
        # return Ifstatement with block as condition
        condition = tail[0]
        parsedStatement = parseIf(tail[1:])
        if type(parsedStatement) == Error:
            return parsedStatement
        else:
            parsedStatement[0].condition = condition
            return parsedStatement

    elif head.type == 'TRUE' and type(tail[0]) == Block:
        # return Ifstatement with block as ifTrue
        if len(tail) > 2 and tail[1].type == 'FALSE':
            falseStatement, nextTail = parseIf(tail[1:])
            result = IfStatement(Block([]), tail[0], falseStatement.ifFalse), nextTail

        else:
            result = IfStatement(Block([]), tail[0]), tail[1:]

    elif head.type == 'FALSE' and type(tail[0]) == Block:
        # return Ifstatement with block as ifFalse
        if len(tail) > 2 and tail[1].type == 'TRUE':
            trueStatement, nextTail = parseIf(tail[1:])
            result = IfStatement(Block([]), trueStatement.ifTrue, tail[0]), nextTail
        else:
            result = IfStatement(Block([]), ifFalse=tail[0]), tail[1:]

    # if if token is followed by a block token
    elif head.type == 'IF' and tail[0].type == 'BLOCK':
        # parse that block as condition and call parseIfStatement again to parse the ifTrue and/or ifFalse
        parsedBlock = parseBlock(tail)
        condition = parsedBlock[0]

        parsedStatement = parseIf(parsedBlock[1])
        if parsedStatement == None:
            return Error('Expected yea or nah block after if statement', head.line, head.position)

        parsedStatement: Union[Error, Tuple[IfStatement, List[Token]]]
        if type(parsedStatement) == Error:
            result = parsedStatement
        else:
            parsedStatement[0].condition = condition
            result = parsedStatement

    elif head.type == 'IF' and (tail[0].type != 'BLOCK' or type(tail[0] != Block)):
        result = Error('if statement needs to be followed by a code block', head.line, head.position)

    # if true token is followed by a block token
    elif head.type == 'TRUE' and tail[0].type == 'BLOCK':
        # parse that block as ifTrue
        parsedBlock = parseBlock(tail)
        ifTrue = parsedBlock[0]
        nextTail = parsedBlock[1]
        # Check if it is followed by an ifFalse statement
        if len(nextTail) and nextTail[0].type == 'FALSE':
            parsedFalseStatement = parseIf(nextTail)
            if type(parsedFalseStatement) == Error:
                result = parsedFalseStatement
            elif parsedFalseStatement[0].ifTrue != None:
                result = Error('If statement can only have 1 yea block', head.line, head.position)
            else:
                result = IfStatement(Block([]), ifTrue, parsedFalseStatement[0].ifFalse), parsedFalseStatement[1]

        elif len(nextTail) and nextTail[0].type == 'TRUE':
            result = Error('If statement can only have 1 yea block', head.line, head.position)

        else:
            result = IfStatement(Block([]), ifTrue), nextTail

    # if false token is followed by a block token
    elif head.type == 'FALSE' and tail[0].type == 'BLOCK':
        # parse that block as ifFalse
        parsedBlock = parseBlock(tail)
        ifFalse = parsedBlock[0]
        nextTail = parsedBlock[1]
        if len(nextTail) and nextTail[0].type == 'TRUE':
            # Check if it is followed by an ifTrue statement
            parsedTrueStatement = parseIf(nextTail)
            if type(parsedTrueStatement) == Error:
                result = parsedTrueStatement
            elif parsedTrueStatement[0].ifFalse != None:
                result = Error('If statement can only have 1 nah block', head.line, head.position)
            else:
                result = IfStatement(Block([]), parsedTrueStatement[0].ifTrue, ifFalse), parsedTrueStatement[1]

        elif len(nextTail) and nextTail[0].type == 'FALSE':
            result = Error('If statement can only have 1 nah block', head.line, head.position)
        else:
            result = IfStatement(Block([]), ifFalse=ifFalse), nextTail

    else:
        result = None

    return result

# parsePrint :: List[Token] -> Union[Error, Tuple[AST, List[Token]]]
def parsePrint(tokenList: List[Token]) -> Union[Error, Tuple[AST, List[Token]]]:
    head, *tail = tokenList
    if head.type == 'PRINT':
        if type(tail[0]) == Block:
            test = PrintStatement(tail[0]), tail[1:]
            return test
        elif tail[0].type == 'BLOCK':
            parsedBlock, nextTail = parseBlock(tail)
            return PrintStatement(parsedBlock), nextTail
    else:
        return Error('something went wrong with print', head.line, head.position)

# parseExpression :: List[Token] -> List[AST] -> Union[Error, Tuple[AST, List[Token]]]
def parseExpression(tokenList: List[Token], last: List[AST] = None) -> Union[Error, Tuple[AST, List[Token]]]:
    if last == None:
        last = []
    prev = deepcopy(last)

    if len(tokenList) > 0:
        # check for type and call corresponding parse function
        head, *tail = tokenList

        if type(head) == Block:
            result = head, tail

        elif head.type == 'NUMBER' and not len(prev):
            prev.append(Number(head))
            result = parseExpression(tail, prev)
        elif head.type == 'NUMBER' and prev != []:
            result = Error('did not expect ' + str(tail[0].value) + ' before number', head.value.line, head.value.position)

        elif head.type == 'IDENTIFIER':
            if not len(prev):
                prev.append(Identifier(head))
                expression = parseExpression(tail, prev)
                result = expression
                if type(result[0]) == list:
                    result = result[0][0], result[1]
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
            else:
                result = Error('Expected variable before assignment', head.line, head.position)

        elif head.type == 'IF':
            result = parseIf(tokenList)

        elif head.type == 'TRUE' or head.type == 'FALSE':
            result = Error('Expected if-statement before code block', head.line, head.position)

        elif head.type == 'WHILE':
            result = parseWhile(tokenList)

        elif head.type == 'EXECUTE':
            result = Error('Expected while-statement before execute', head.line, head.position)

        elif head.type == 'BLOCK':
            result = parseBlock(tokenList)

        elif head.type == 'PRINT':
            result = parsePrint(tokenList)

        elif head.type == 'END':
            if len(prev):
                result = prev, tail

            else:
                result = parseExpression(tail)

        else:
            result = Error('Unhandled token: ' + head, head.line, head.position)

    else:
        result = 'EOF'

    return result

# parse :: List[Token] -> Union[Error, Tuple[AST, List[Token]], Tuple[AST, str] ]
def parse(tokenList: List[Token]) -> Union[Error, Tuple[AST, List[Token]], Tuple[AST, str] ]:
    result: Tuple[AST, List[Token]] = parseExpression(tokenList)
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

# Class to represent the program state
class State:
    def __init__(self):
        self.variables = {}
        self.errors = []
    def __repr__(self):
        return 'State(Variables: ' + str(self.variables) + ', errors: ' + str(self.errors) +  ')'

# visitNumber :: Number -> State -> Tuple[int, State]
def visitNumber(node: Number, originalState: State) -> Tuple[int, State]:
    return int(node.value.value), originalState

# visitIdentifier :: Identifier -> State -> Union[Tuple[None, State], Tuple[int, State]]
def visitIdentifier(node: Identifier, originalState: State) -> Union[Tuple[None, State], Tuple[int, State]]:
    newState = deepcopy(originalState)
    variableValue = originalState.variables.get(node.value.value)
    # return value if variable name exists, otherwise add error to state
    if variableValue == None:
        newState.errors.append(Error('Expected a value', node.value.line, node.value.position))
        return None, newState
    else:
        return variableValue, newState

# visitAssign :: Assign -> State -> State
def visitAssign(node: Assign, originalState: State) -> State:
    value = visit(node.value, originalState)
    if value[0] == None:
        # if the visited value does not return anything, return the newState
        newState = value[1]
        return newState
    else:
        # update newState with variable and return the newState
        newState = originalState
        newState.variables.update({node.variable.value.value : value[0]})
        return newState

# visitBinaryOperator :: BinaryOperator -> State -> Tuple[int, State]
def visitBinaryOperator(node: BinaryOperator, originalState: State) -> Union[State, Tuple[Union[int, float], State]]:
    newState = deepcopy(originalState)
    lhs = visit(node.left, originalState)[0]
    rhs = visit(node.right, originalState)[0]
    operators = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "<": operator.lt,
        ">": operator.gt,
        "eq": operator.eq
    }
    operatorFunction = operators[node.operator.value]
    if (type(lhs) == int or type(lhs) == float) and (type(rhs) == int or type(rhs) == float):
        return operatorFunction(lhs, rhs), newState
    else:
        newState.errors.append(Error('Expected Number before operator' , node.operator.line, node.operator.position))
        return newState

# visitWhileStatement :: WhileStatement -> State -> Union[State, Tuple[int, State]]
def visitWhileStatement(node: WhileStatement, originalState: State) -> Union[State, Tuple[int, State]]:
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

# visitIfSTatement :: IfStatement -> State -> State
def visitIfStatement(node: IfStatement, originalState: State) -> State:
    condition = visit(node.condition, originalState)
    if type(condition) == Error:
        newState = deepcopy(originalState)
        newState.errors.append(condition)
        result = newState
    elif condition[0] == True:
        result = visit(node.ifTrue, condition[1])
    else:
        if node.ifFalse != None:
            result = visit(node.ifFalse, condition[1])
        else:
            result = condition[1]
    if type(result) == State:
        return result
    else:
        return result[-1]

# visitBlock :: Block -> State -> Union[State, Tuple[int, State], Tuple[Tuple[int], State]]
def visitBlock(node: Block, originalState: State) -> Union[State, Tuple[int, State], Tuple[Tuple[int], State]]:
    # visit each item in block
    head, *tail = node.expressions
    firstNode = visit(head, originalState)
    if len(tail) > 0:
        if type(firstNode) == State:
            nextParsedBlock = visit(Block(tail), firstNode)
            return nextParsedBlock
        else:
            nextParsedBlock = visit(Block(tail), firstNode[1])
            if type(nextParsedBlock) == State:
                result = (firstNode[0],) + (nextParsedBlock,)
            else:
                result = (firstNode[0],) + nextParsedBlock
            return result

    return firstNode

# visitPrintStatement :: PrintStatement -> State -> State
def visitPrintStatement(node: PrintStatement, originalState: State) -> State:
    result = visit(node.result, originalState)
    if type(result) != State:
        print(result[0])
    return originalState


# visit :: AST -> State -> Union[State, Tuple[int, State], Tuple[Tuple[int], State] ]
def visit(node: AST, originalState: State) -> Union[State, Tuple[int, State], Tuple[Tuple[int], State] ]:
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
    elif type(node) == PrintStatement:
        node: PrintStatement
        return visitPrintStatement(node, originalState)
    else:
        newState = deepcopy(originalState)
        newState.errors.append(GlobalError('Unknown node is called: ' + str(node)))
        return newState

# interpret :: List[AST] -> State -> Union[State, Tuple[int, State]]
def interpret(ast: Union[Error, List[AST]], originalState: State) -> Union[State, Tuple[int, State]]:
    newState = deepcopy(originalState)
    if type(ast) == Error:
        newState.errors.append(ast)
        return newState

    head, *tail = ast

    if tail != 'EOF' and len(tail) > 1:
        currentExpression = visit(head, newState)
        if type(currentExpression) == State:
            return interpret(tail, currentExpression)
        else:
            nextExpression = interpret(tail, currentExpression[-1])
            if type(nextExpression) == State:
                return currentExpression[:-1] + (nextExpression,)
            else:
                return currentExpression[:-1] + nextExpression
    elif tail[0] == 'EOF':
        return visit(head, newState)
    else:
        newState.errors.append(GlobalError('Something went wrong'))
        return newState


# ---------------------------------------------
# Run/Debug
# ---------------------------------------------

def run():
    if len(sys.argv) == 1:
        inputFile = open("input.txt", "r")
        sourceCode = inputFile.read()
    elif len(sys.argv) > 1 and sys.argv[1].endswith('.txt'):
        inputFile = open(sys.argv[1], "r")
        sourceCode = inputFile.read()
    else:
        test = ' '
        sourceCode = test.join(sys.argv[1:])


    originalState = State()
    programResult = interpret(parse(lex(sourceCode)), originalState)
    if type(programResult) == State:
        programState = programResult
    else:
        *returnedValues, programState = programResult
    print(programState)


    # parse(lex(sourceCode))
    # print(parse(lex(sourceCode)))
    # print(interpret.getStats())
    # print(visit.getStats())
    # print(parsePrint.getStats())
    # print(parseWhile.getStats())
# Reading input file
# inputFile = open("input.txt", "r")
# sourceCode = inputFile.read()
run()




import inspect
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generator, List, NoReturn, Callable, Dict, TextIO, Tuple, Optional, BinaryIO


@dataclass
class Loc:
    filename: str
    line: int
    col: int


def asm_setup(write_base: Callable[[str], None], write_level1: Callable[[str], None]) -> None:
    write_base('.macro push reg1:req')
    write_level1(r'str \reg1, [sp, #-16]!')
    write_base('.endmacro')
    write_base('.macro pop reg1:req')
    write_level1(r'ldr \reg1, [sp], #16')
    write_base('.endmacro')
    write_base('print:')
    write_level1('sub     sp, sp,  #48')
    write_level1('stp     x29, x30, [sp,  #32]')
    write_level1('add     x29, sp,  #32')
    write_level1('mov     x10,  #-3689348814741910324')
    write_level1('mov     x8, xzr')
    write_level1('mov     w9,  #10')
    write_level1('movk    x10,  #52429')
    write_level1('mov     x11, sp')
    write_level1('strb    w9, [sp,  #31]')
    write_base('.LBB0_1:')
    write_level1('umulh   x12, x0, x10')
    write_level1('add     x14, x11, x8')
    write_level1('sub     x8, x8,  #1')
    write_level1('cmp     x0,  #9')
    write_level1('lsr     x12, x12,  #3')
    write_level1('msub    w13, w12, w9, w0')
    write_level1('mov     x0, x12')
    write_level1('orr     w13, w13,  #0x30')
    write_level1('strb    w13, [x14,  #30]')
    write_level1('b.hi    .LBB0_1')
    write_level1('mov     x9, sp')
    write_level1('mov     w10,  #1')
    write_level1('add     x9, x9, x8')
    write_level1('sub     x2, x10, x8')
    write_level1('add     x1, x9,  #31')
    write_level1('mov     x0,  #1')
    write_level1('mov     x16,  #4')
    write_level1('svc  #0')
    write_level1('ldp     x29, x30, [sp,  #32]')
    write_level1('add     sp, sp,  #48')
    write_level1('ret')


class Intrinsic(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    PRINT = auto()
    EQUAL = auto()
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    NE = auto()
    DUP = auto()
    DROP = auto()
    AND = auto()
    OR = auto()
    SHIFT_RIGHT = auto()
    SHIFT_LEFT = auto()
    SWAP = auto()
    OVER = auto()
    ROT = auto()
    MEM = auto()
    LOAD = auto()
    STORE = auto()
    LOAD64 = auto()
    STORE64 = auto()
    SYSCALL1 = auto()
    SYSCALL3 = auto()
    ARGC = auto()
    ARGV = auto()
    CAST_PTR = auto()
    HERE = auto()

class OpType(Enum):
    PUSH_INT = auto()
    PUSH_STR = auto()
    INTRINSIC = auto()
    IF = auto()
    END = auto()
    ELSE = auto()
    WHILE = auto()
    DO = auto()


class Keyword(Enum):
    IF = auto()
    END = auto()
    ELSE = auto()
    WHILE = auto()
    DO = auto()
    MACRO = auto()
    INCLUDE = auto()


class TokenType(Enum):
    WORD = auto()
    KEYWORD = auto()
    INT = auto()
    STR = auto()
    CHAR = auto()


class DataType(Enum):
    INT = auto()
    PTR = auto()
    BOOL = auto()


@dataclass
class Token:
    type: TokenType
    value: int | str | Keyword
    loc: Loc
    name: str
    tokens: list['Token'] | None = None
    expanded_from: Optional['Token'] = None
    expanded_count: int = 0


@dataclass
class Op:
    type: OpType
    token: Token
    name: str
    operand: int | str | Intrinsic | None = None
    addr: int | None = None


KEYWORD_NAMES = {
    'if': Keyword.IF,
    'end': Keyword.END,
    'else': Keyword.ELSE,
    'while': Keyword.WHILE,
    'do': Keyword.DO,
    'macro': Keyword.MACRO,
    'include': Keyword.INCLUDE,
}

assert len(KEYWORD_NAMES) == len(Keyword), 'Exhaustive handling of keywords'

INTRINSIC_NAMES = {
    'print': Intrinsic.PRINT,
    '+': Intrinsic.ADD,
    '-': Intrinsic.SUB,
    '*': Intrinsic.MUL,
    '/': Intrinsic.DIV,
    '==': Intrinsic.EQUAL,
    '>': Intrinsic.GT,
    '<': Intrinsic.LT,
    'dup': Intrinsic.DUP,
    'drop': Intrinsic.DROP,
    'and': Intrinsic.AND,
    'or': Intrinsic.OR,
    '>>': Intrinsic.SHIFT_RIGHT,
    '<<': Intrinsic.SHIFT_LEFT,
    'swap': Intrinsic.SWAP,
    'over': Intrinsic.OVER,
    'rot': Intrinsic.ROT,
    '>=': Intrinsic.GTE,
    '<=': Intrinsic.LTE,
    '!=': Intrinsic.NE,
    'mem': Intrinsic.MEM,
    '.': Intrinsic.STORE,
    ',': Intrinsic.LOAD,
    '.64': Intrinsic.STORE64,
    ',64': Intrinsic.LOAD64,
    'syscall1': Intrinsic.SYSCALL1,
    'syscall3': Intrinsic.SYSCALL3,
    'argc': Intrinsic.ARGC,
    'argv': Intrinsic.ARGV,
    'int->ptr': Intrinsic.CAST_PTR,
    'here': Intrinsic.HERE,
}

assert len(INTRINSIC_NAMES) == len(Intrinsic), 'Exhaustive handling of intrinsics'

MACRO_EXPANSION_LIMIT = 100_000
MACRO_TRACEBACK_LIMIT = 10
NULL_POINTER_PADDING = 1  # padding to make 0 an invalid address
ARG_PTR_CAPACITY = 640 + NULL_POINTER_PADDING
STR_CAPACITY = 640_000 + ARG_PTR_CAPACITY
MEM_CAPACITY = 640_000 + STR_CAPACITY
FDS: List[BinaryIO] = [sys.stdin.buffer, sys.stdout.buffer, sys.stderr.buffer]


def make_log_message(message: str, loc: Loc) -> str:
    return f'{loc.filename}:[{loc.line + 1}:{loc.col}]: {message}'


def notify_user(message: str, loc: Loc) -> None:
    print(make_log_message('[NOTE] ' + message, loc))


def traceback_message(frame: int = 1) -> None:
    current_frame = inspect.currentframe()
    caller_frame = inspect.getouterframes(current_frame)
    caller_info = caller_frame[frame]
    caller_function = caller_info.function
    trace_message = f'Error message originated inside: {caller_function}'
    print(make_log_message('[ERROR] ' + trace_message, Loc(filename='./photon.py',
                                                           line=caller_info.lineno - 1,
                                                           col=0)), file=sys.stderr)


def raise_error(message: str, place: Loc | Token, frame: int = 2) -> NoReturn:
    traceback_message(frame=frame)
    if isinstance(place, Token):
        i = 0
        expanded_count = place.expanded_count
        expand_place = place
        while i < MACRO_TRACEBACK_LIMIT and i < expanded_count:
            assert expand_place.expanded_from is not None, 'Bug in macro expansion count'
            notify_user(f'Operation expanded from macro: {expand_place.expanded_from.value}',
                        loc=expand_place.expanded_from.loc)
            expand_place = expand_place.expanded_from
            i += 1
        place = place.loc
    print(make_log_message('[ERROR] ' + message, place), file=sys.stderr)
    exit(1)


def get_cstr_from_mem(mem: bytearray, ptr: int) -> bytes:
    end = ptr
    while mem[end] != 0:
        end += 1
    return mem[ptr:end]


def write_indent(file: TextIO, level: int = 0) -> Callable[[str], None]:
    def temp(buffer: str) -> None:
        file.write(' ' * (0 if level == 0 else 4 ** level) + buffer + '\n')

    return temp


def ensure_argument_count(stack_length: int, op: Op, required: int) -> None | NoReturn:
    if stack_length < required:
        traceback_message(2)
        raise_error(f'Not enough arguments for: {op.name}, found: {stack_length} but required: {required}',
                    op.token)
    return None


def notify_argument_origin(loc: Token, order: int = 1) -> None:
    notify_user(f'Argument {order} was created at this location', loc.loc)


DataTypeStack = List[Tuple[DataType, Token]]


def type_check_program(program: List[Op], debug: bool = False) -> None:
    stack: DataTypeStack = []
    block_stack: List[Tuple[DataTypeStack, Op]] = []  # convert stack to tuple keeping only DataType, hash and store
    for op in program:
        assert len(OpType) == 8, 'Exhaustive handling of operations in type check'
        if op.type == OpType.PUSH_INT:
            assert type(op.operand) == int, 'Value for `PUSH_INT` must be `int`'
            stack.append((DataType.INT, op.token))
        elif op.type == OpType.PUSH_STR:
            assert type(op.operand) == str, 'Value for `PUSH_STR` must be `str`'
            stack.append((DataType.INT, op.token))
            stack.append((DataType.PTR, op.token))
        elif op.type == OpType.IF:
            ensure_argument_count(len(stack), op, 1)
            a_type, a_loc = stack.pop()
            if a_type != DataType.BOOL:
                if debug:
                    notify_argument_origin(a_loc, order=1)
                raise_error(f'Invalid argument types for `{op.name}`: {a_type.name}', op.token)
            block_stack.append((stack.copy(), op))
        elif op.type == OpType.ELSE:
            before_if_stack, if_op = block_stack.pop()
            assert if_op.type == OpType.IF, '[BUG] else without if'
            block_stack.append((stack.copy(), op))
            stack = before_if_stack
        elif op.type == OpType.END:
            stack_before_block, block = block_stack.pop()
            expected_stack = list(map(lambda x: x[0], stack_before_block))
            current_stack = list(map(lambda x: x[0], stack))
            if block.type == OpType.IF:
                if current_stack != expected_stack:
                    notify_user(f'Expected Stack Types: {expected_stack}', op.token.loc)
                    notify_user(f'Actual Stack Types: {current_stack}', op.token.loc)
                    raise_error('Stack types cannot be altered after an else-less if block', op.token)
            elif block.type == OpType.ELSE:
                if current_stack != expected_stack:
                    notify_user(f'Expected Stack Types: {expected_stack}', op.token.loc)
                    notify_user(f'Actual Stack Types: {current_stack}', op.token.loc)
                    raise_error('Both branches of an if-else block must produce the same stack types', op.token)
            elif block.type == OpType.DO:
                stack_before_while, while_op = block_stack.pop()
                assert while_op.type == OpType.WHILE, '[BUG] No `while` before `do`'
                expected_stack = list(map(lambda x: x[0], stack_before_while))
                if current_stack != expected_stack:
                    notify_user(f'Expected Stack Types: {expected_stack}', op.token.loc)
                    notify_user(f'Actual Stack Types: {current_stack}', op.token.loc)
                    raise_error('Stack types cannot be altered inside a while-do body', op.token)
                stack = stack_before_block
            else:
                assert False, 'Unreachable'
        elif op.type == OpType.WHILE:
            block_stack.append((stack.copy(), op))
        elif op.type == OpType.DO:
            ensure_argument_count(len(stack), op, 1)
            a_type, a_loc = stack.pop()
            if a_type != DataType.BOOL:
                if debug:
                    notify_argument_origin(a_loc, order=1)
                raise_error(f'Invalid argument types for `{op.name}`: {a_type.name}', op.token)
            block_stack.append((stack.copy(), op))
        elif op.type == OpType.INTRINSIC:
            assert len(Intrinsic) == 31, 'Exhaustive handling of intrinsics in type check'
            if op.operand == Intrinsic.ADD:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.INT, op.token))
                elif a_type == DataType.PTR and b_type == DataType.INT:
                    stack.append((DataType.PTR, op.token))
                elif a_type == DataType.INT and b_type == DataType.PTR:
                    stack.append((DataType.PTR, op.token))
                else:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.SUB:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.INT, op.token))
                elif a_type == DataType.INT and b_type == DataType.PTR:
                    stack.append((DataType.PTR, op.token))
                elif a_type == DataType.PTR and b_type == DataType.PTR:
                    stack.append((DataType.INT, op.token))
                else:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.MUL:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.INT, op.token))
                else:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.DIV:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.INT, op.token))
                else:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.PRINT:
                ensure_argument_count(len(stack), op, 1)
                stack.pop()
            elif op.operand == Intrinsic.EQUAL:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.BOOL, op.token))
                elif a_type == DataType.PTR and b_type == DataType.PTR:
                    stack.append((DataType.BOOL, op.token))
                elif a_type == DataType.BOOL and b_type == DataType.BOOL:
                    stack.append((DataType.BOOL, op.token))
                else:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.LT:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.BOOL, op.token))
                elif a_type == DataType.PTR and b_type == DataType.PTR:
                    stack.append((DataType.BOOL, op.token))
                else:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.GT:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.BOOL, op.token))
                elif a_type == DataType.PTR and b_type == DataType.PTR:
                    stack.append((DataType.BOOL, op.token))
                else:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.GTE:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.BOOL, op.token))
                elif a_type == DataType.PTR and b_type == DataType.PTR:
                    stack.append((DataType.BOOL, op.token))
                else:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.LTE:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.BOOL, op.token))
                elif a_type == DataType.PTR and b_type == DataType.PTR:
                    stack.append((DataType.BOOL, op.token))
                else:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.NE:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.BOOL, op.token))
                elif a_type == DataType.PTR and b_type == DataType.PTR:
                    stack.append((DataType.BOOL, op.token))
                elif a_type == DataType.BOOL and b_type == DataType.BOOL:
                    stack.append((DataType.BOOL, op.token))
                else:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.DUP:
                ensure_argument_count(len(stack), op, 1)
                a = stack.pop()
                stack.append(a)
                stack.append(a)
            elif op.operand == Intrinsic.DROP:
                ensure_argument_count(len(stack), op, 1)
                stack.pop()
            elif op.operand == Intrinsic.MEM:
                stack.append((DataType.PTR, op.token))
            elif op.operand == Intrinsic.ARGC:
                stack.append((DataType.INT, op.token))
            elif op.operand == Intrinsic.ARGV:
                stack.append((DataType.PTR, op.token))
            elif op.operand == Intrinsic.LOAD:
                ensure_argument_count(len(stack), op, 1)
                a_type, a_loc = stack.pop()
                if a_type != DataType.PTR:
                    if debug:
                        notify_argument_origin(a_loc, order=1)
                    raise_error(f'Invalid argument types for `{op.name}`: {a_type.name}', op.token)
                stack.append((DataType.INT, op.token))
            elif op.operand == Intrinsic.STORE:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type != DataType.INT or b_type != DataType.PTR:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.LOAD64:
                ensure_argument_count(len(stack), op, 1)
                a_type, a_loc = stack.pop()
                if a_type != DataType.PTR:
                    if debug:
                        notify_argument_origin(a_loc, order=1)
                    raise_error(f'Invalid argument types for `{op.name}`: {a_type.name}', op.token)
                stack.append((DataType.INT, op.token))
            elif op.operand == Intrinsic.STORE64:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type != DataType.INT or b_type != DataType.PTR:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.OR:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.INT, op.token))
                elif a_type == DataType.BOOL and b_type == DataType.BOOL:
                    stack.append((DataType.BOOL, op.token))
                else:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.AND:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.INT, op.token))
                elif a_type == DataType.BOOL and b_type == DataType.BOOL:
                    stack.append((DataType.BOOL, op.token))
                else:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.SHIFT_RIGHT:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.INT, op.token))
                else:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.SHIFT_LEFT:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.INT, op.token))
                else:
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.SWAP:
                ensure_argument_count(len(stack), op, 2)
                a = stack.pop()
                b = stack.pop()
                stack.append(a)
                stack.append(b)
            elif op.operand == Intrinsic.OVER:
                ensure_argument_count(len(stack), op, 2)
                a = stack.pop()
                b = stack.pop()
                stack.append(b)
                stack.append(a)
                stack.append(b)
            elif op.operand == Intrinsic.ROT:
                ensure_argument_count(len(stack), op, 3)
                a = stack.pop()
                b = stack.pop()
                c = stack.pop()
                stack.append(b)
                stack.append(a)
                stack.append(c)
            elif op.operand == Intrinsic.CAST_PTR:
                ensure_argument_count(len(stack), op, 1)
                a_type, a_loc = stack.pop()
                if a_type != DataType.INT:
                    if debug:
                        notify_argument_origin(a_loc, order=1)
                    raise_error(f'Invalid argument types for `{op.name}`: {a_type.name}', op.token)
                stack.append((DataType.PTR, op.token))
            elif op.operand == Intrinsic.HERE:
                stack.append((DataType.INT, op.token))
                stack.append((DataType.PTR, op.token))
            elif op.operand == Intrinsic.SYSCALL1:
                ensure_argument_count(len(stack), op, 2)
                syscall_type, syscall_loc = stack.pop()
                arg1_type, arg1_loc = stack.pop()
                if syscall_type != DataType.INT or arg1_type != DataType.INT:
                    if debug:
                        notify_argument_origin(arg1_loc, order=1)
                        notify_argument_origin(syscall_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(arg1_type.name, syscall_type.name)}',
                                op.token)
                stack.append((DataType.INT, op.token))
            elif op.operand == Intrinsic.SYSCALL3:
                ensure_argument_count(len(stack), op, 4)
                syscall_type, syscall_loc = stack.pop()
                arg3_type, arg3_loc = stack.pop()
                arg2_type, arg2_loc = stack.pop()
                arg1_type, arg1_loc = stack.pop()
                if syscall_type != DataType.INT or arg1_type != DataType.INT:
                    if debug:
                        notify_argument_origin(arg1_loc, order=1)
                        notify_argument_origin(arg2_loc, order=2)
                        notify_argument_origin(arg3_loc, order=3)
                        notify_argument_origin(syscall_loc, order=4)
                    raise_error(
                        f'Invalid argument types for `{op.name}`: '
                        f'{(arg1_type.name, arg2_type.name, arg3_type.name, syscall_type.name)}',
                        op.token)
                stack.append((DataType.INT, op.token))
            else:
                raise_error(f'Unhandled intrinsic: {op.name}',
                            op.token.loc)
        else:
            raise_error(f'Unhandled op: {op.name}',
                        op.token.loc)
    if len(stack) != 0:
        current_stack = list(map(lambda x: x[0], stack))
        raise_error(f'Unhandled data on the stack: {current_stack}', stack[-1][1])


def simulate_little_endian_macos(program: List[Op], input_arguments: List[str]) -> None:
    stack: List = []
    assert len(OpType) == 8, 'Exhaustive handling of operators in simulation'
    i = 0
    mem = bytearray(MEM_CAPACITY)
    allocated_strs = {}
    ptr_size = NULL_POINTER_PADDING
    str_size = ARG_PTR_CAPACITY
    for arg in input_arguments:
        arg_value = arg.encode('utf-8')
        n = len(arg_value)
        mem[str_size:str_size + n] = arg_value
        mem[str_size + n] = 0
        mem[ptr_size:ptr_size + 8] = str_size.to_bytes(8, 'little')
        ptr_size += 8
        str_size += n + 1
        assert str_size <= STR_CAPACITY, "String buffer overflow"
        assert ptr_size <= ARG_PTR_CAPACITY, "Argument pointer buffer overflow"
    argc = len(input_arguments)
    argv_start = NULL_POINTER_PADDING
    while i < len(program):
        op = program[i]
        if op.type == OpType.PUSH_INT:
            assert type(op.operand) == int, 'Value for `PUSH_INT` must be `int`'
            stack.append(op.operand)
        elif op.type == OpType.PUSH_STR:
            assert type(op.operand) == str, 'Value for `PUSH_STR` must be `str`'
            bs = bytes(op.operand, 'utf-8')
            n = len(bs)
            stack.append(n)
            if op.operand not in allocated_strs:
                allocated_strs[op.operand] = str_size
                mem[str_size:str_size + n] = bs
                str_size += n
                if str_size > STR_CAPACITY:
                    raise_error('String buffer overflow', op.token.loc)
            stack.append(allocated_strs[op.operand])
        elif op.type == OpType.IF:
            a = stack.pop()
            if a == 0:
                assert type(op.operand) == int, 'Jump address must be `int`'
                i = op.operand
        elif op.type == OpType.ELSE:
            assert type(op.operand) == int, 'Jump address must be `int`'
            i = op.operand
        elif op.type == OpType.END:
            if op.operand is not None:
                assert type(op.operand) == int, 'Jump address must be `int`'
                i = op.operand
        elif op.type == OpType.WHILE:
            i += 1
            continue
        elif op.type == OpType.DO:
            a = stack.pop()
            assert type(a) == int, 'Arguments for `do` must be `int`'
            if a == 0:
                assert type(op.operand) == int, 'Jump address must be `int`'
                i = op.operand
        elif op.type == OpType.INTRINSIC:
            assert len(Intrinsic) == 31, 'Exhaustive handling of intrinsics in simulation'
            if op.operand == Intrinsic.ADD:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `+` must be `int`'
                stack.append(a + b)
            elif op.operand == Intrinsic.SUB:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `-` must be `int`'
                stack.append(b - a)
            elif op.operand == Intrinsic.MUL:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `*` must be `int`'
                stack.append(a * b)
            elif op.operand == Intrinsic.DIV:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `/` must be `int`'
                stack.append(b // a)
            elif op.operand == Intrinsic.PRINT:
                a = stack.pop()
                assert type(a) == int, 'Arguments for `print` must be `int`'
                print(a)
            elif op.operand == Intrinsic.EQUAL:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `==` must be `int`'
                stack.append(int(a == b))
            elif op.operand == Intrinsic.LT:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `<` must be `int`'
                stack.append(int(b < a))
            elif op.operand == Intrinsic.GT:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `>` must be `int`'
                stack.append(int(b > a))
            elif op.operand == Intrinsic.GTE:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `>=` must be `int`'
                stack.append(int(b >= a))
            elif op.operand == Intrinsic.LTE:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `<=` must be `int`'
                stack.append(int(b <= a))
            elif op.operand == Intrinsic.NE:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `!=` must be `int`'
                stack.append(int(a != b))
            elif op.operand == Intrinsic.DUP:
                a = stack.pop()
                assert type(a) == int, 'Arguments for `dup` must be `int`'
                stack.append(a)
                stack.append(a)
            elif op.operand == Intrinsic.DROP:
                stack.pop()
            elif op.operand == Intrinsic.MEM:
                stack.append(STR_CAPACITY)
            elif op.operand == Intrinsic.ARGC:
                stack.append(argc)
            elif op.operand == Intrinsic.ARGV:
                stack.append(argv_start)
            elif op.operand == Intrinsic.LOAD:
                address = stack.pop()
                assert type(address) == int, 'Arguments for `,` must be `int`'
                stack.append(mem[address])
            elif op.operand == Intrinsic.STORE:
                value = stack.pop()
                address = stack.pop()
                assert type(value) == type(address) == int, 'Arguments for `.` must be `int`'
                mem[address] = value & 0xFF
            elif op.operand == Intrinsic.LOAD64:
                addr = stack.pop()
                _bytes = bytearray(8)
                for offset in range(0, 8):
                    _bytes[offset] = mem[addr + offset]
                stack.append(int.from_bytes(_bytes, byteorder="little"))
            elif op.operand == Intrinsic.STORE64:
                store_value64 = stack.pop().to_bytes(length=8, byteorder="little")
                store_addr64 = stack.pop()
                for byte in store_value64:
                    mem[store_addr64] = byte
                    store_addr64 += 1
            elif op.operand == Intrinsic.OR:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `or` must be `int`'
                stack.append(a | b)
            elif op.operand == Intrinsic.AND:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `and` must be `int`'
                stack.append(a & b)
            elif op.operand == Intrinsic.SHIFT_RIGHT:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `>>` must be `int`'
                stack.append(b >> a)
            elif op.operand == Intrinsic.SHIFT_LEFT:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `<<` must be `int`'
                stack.append(b << a)
            elif op.operand == Intrinsic.SWAP:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `swap` must be `int`'
                stack.append(a)
                stack.append(b)
            elif op.operand == Intrinsic.OVER:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `over` must be `int`'
                stack.append(b)
                stack.append(a)
                stack.append(b)
            elif op.operand == Intrinsic.ROT:
                a = stack.pop()
                b = stack.pop()
                c = stack.pop()
                assert type(a) == type(b) == type(c) == int, 'Arguments for `over` must be `int`'
                stack.append(b)
                stack.append(a)
                stack.append(c)
            elif op.operand == Intrinsic.CAST_PTR:
                i += 1
                continue
            elif op.operand == Intrinsic.HERE:
                here_text = f'{op.token.loc.filename}:[{op.token.loc.line + 1}:{op.token.loc.col}]: '
                bs = bytes(here_text, 'utf-8')
                n = len(bs)
                stack.append(n)
                if here_text not in allocated_strs:
                    allocated_strs[here_text] = str_size
                    mem[str_size:str_size + n] = bs
                    str_size += n
                    if str_size > STR_CAPACITY:
                        raise_error('String buffer overflow', op.token.loc)
                stack.append(allocated_strs[here_text])
            elif op.operand == Intrinsic.SYSCALL1:
                syscall_number = stack.pop()
                arg1 = stack.pop()
                if syscall_number == 1:
                    exit(arg1)
                elif syscall_number == 6:
                    try:
                        FDS[arg1].close()
                        stack.append(0)
                    except Exception:
                        stack.append(-1)
                else:
                    assert False, 'Unhandled syscall number'
            elif op.operand == Intrinsic.SYSCALL3:
                syscall_number = stack.pop()
                arg1 = stack.pop()
                arg2 = stack.pop()
                arg3 = stack.pop()
                assert type(arg1) == type(arg2) == type(arg3) == int, 'Arguments for `syscall3` must be `int`'
                if syscall_number == 3:
                    data = FDS[arg1].readline(arg3)
                    mem[arg2:arg2+len(data)] = data
                    stack.append(len(data))
                elif syscall_number == 4:
                    FDS[arg1].write(mem[arg2:arg2 + arg3])
                    FDS[arg1].flush()
                    stack.append(arg3)
                elif syscall_number == 5:
                    flags = ['rb', 'wb', 'wb+']
                    pathname = get_cstr_from_mem(mem, arg1).decode()
                    fdi = len(FDS)
                    if flags[arg2] == 'rb':
                        try:
                            FDS.append(open(pathname, 'rb'))
                        except FileNotFoundError:
                            fdi = -1
                    else:
                        raise_error('Unsupported flag for file access', op.token)
                    stack.append(fdi)
                else:
                    raise_error(f'Unknown syscall number: {syscall_number}', op.token.loc)
            else:
                raise_error(f'Unhandled intrinsic: {op.name}',
                            op.token.loc)
        else:
            raise_error(f'Unhandled operation: {op.name}',
                        op.token.loc)
        i += 1


def compile_program(program: List[Op]) -> None:
    assert len(OpType) == 8, 'Exhaustive handling of operators in compilation'
    out = open('output.s', 'w')
    write_base = write_indent(out, 0)
    write_level1 = write_indent(out, 1)
    write_base('.section __TEXT, __text')
    write_base('.global _main')
    write_base('.align 3')
    asm_setup(write_base, write_level1)
    write_base('_main:')
    write_level1('adrp x2, argc@PAGE')
    write_level1('add x2, x2, argc@PAGEOFF')
    write_level1('str x0, [x2]')
    write_level1('adrp x2, argv@PAGE')
    write_level1('add x2, x2, argv@PAGEOFF')
    write_level1('str x1, [x2]')
    strs: List[str] = []
    allocated_strs: Dict[str, int] = {}
    for i in range(len(program)):
        op = program[i]
        if op.type == OpType.PUSH_INT:
            assert type(op.operand) == int, 'Operation value must be an `int` for PUSH_INT'
            write_level1(f'ldr x0, ={op.operand}')
            write_level1('push x0')
        elif op.type == OpType.PUSH_STR:
            assert type(op.operand) == str, 'Operation value must be a `str` for PUSH_STR'
            write_level1(f'ldr x0, ={len(op.operand)}')
            write_level1('push x0')
            address = allocated_strs.get(op.operand, len(strs))
            write_level1(f'adrp x1, str_{address}@PAGE')
            write_level1(f'add x1, x1, str_{address}@PAGEOFF')
            write_level1('push x1')
            if op.operand not in allocated_strs:
                allocated_strs[op.operand] = len(strs)
                strs.append(op.operand)
        elif op.type in (OpType.IF, OpType.DO):
            write_level1('pop x0')
            write_level1('tst x0, x0')
            write_level1(f'b.eq end_{op.operand}')
        elif op.type == OpType.ELSE:
            write_level1(f'b end_{op.operand}')
            write_base(f'end_{i}:')
        elif op.type == OpType.END:
            if op.operand is not None:
                write_level1(f'b while_{op.operand}')
            write_base(f'end_{i}:')
        elif op.type == OpType.WHILE:
            write_base(f'while_{i}:')
        elif op.type == OpType.INTRINSIC:
            assert len(Intrinsic) == 31, 'Exhaustive handling of intrinsics in simulation'
            if op.operand == Intrinsic.ADD:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('add x0, x0, x1')
                write_level1('push x0')
            elif op.operand == Intrinsic.SUB:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('sub x0, x1, x0')
                write_level1('push x0')
            elif op.operand == Intrinsic.MUL:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('mul x0, x0, x1')
                write_level1('push x0')
            elif op.operand == Intrinsic.DIV:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('udiv x0, x1, x0')
                write_level1('push x0')
            elif op.operand == Intrinsic.PRINT:
                write_level1('pop x0')
                write_level1('bl print')
            elif op.operand == Intrinsic.EQUAL:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('cmp x0, x1')
                write_level1('cset x0, eq')
                write_level1('push x0')
            elif op.operand == Intrinsic.LT:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('cmp x1, x0')
                write_level1('cset x0, lt')
                write_level1('push x0')
            elif op.operand == Intrinsic.GT:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('cmp x1, x0')
                write_level1('cset x0, gt')
                write_level1('push x0')
            elif op.operand == Intrinsic.LTE:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('cmp x1, x0')
                write_level1('cset x0, le')
                write_level1('push x0')
            elif op.operand == Intrinsic.GTE:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('cmp x1, x0')
                write_level1('cset x0, ge')
                write_level1('push x0')
            elif op.operand == Intrinsic.NE:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('cmp x0, x1')
                write_level1('cset x0, ne')
                write_level1('push x0')
            elif op.operand == Intrinsic.DUP:
                write_level1('pop x0')
                write_level1('push x0')
                write_level1('push x0')
            elif op.operand == Intrinsic.DROP:
                write_level1('pop x0')
            elif op.operand == Intrinsic.MEM:
                write_level1('adrp x0, mem@PAGE')
                write_level1('add x0, x0, mem@PAGEOFF')
                write_level1('push x0')
            elif op.operand == Intrinsic.ARGC:
                write_level1('adrp x0, argc@PAGE')
                write_level1('add x0, x0, argc@PAGEOFF')
                write_level1('ldr x0, [x0]')
                write_level1('push x0')
            elif op.operand == Intrinsic.ARGV:
                write_level1('adrp x0, argv@PAGE')
                write_level1('add x0, x0, argv@PAGEOFF')
                write_level1('ldr x0, [x0]')
                write_level1('push x0')
            elif op.operand == Intrinsic.STORE:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('strb w0, [x1]')
            elif op.operand == Intrinsic.LOAD:
                write_level1('pop x0')
                write_level1('ldrb w1, [x0]')
                write_level1('push x1')
            elif op.operand == Intrinsic.STORE64:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('str x0, [x1]')
            elif op.operand == Intrinsic.LOAD64:
                write_level1('pop x0')
                write_level1('ldr x1, [x0]')
                write_level1('push x1')
            elif op.operand == Intrinsic.OR:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('orr x0, x0, x1')
                write_level1('push x0')
            elif op.operand == Intrinsic.AND:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('and x0, x0, x1')
                write_level1('push x0')
            elif op.operand == Intrinsic.SHIFT_RIGHT:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('lsr x0, x1, x0')
                write_level1('push x0')
            elif op.operand == Intrinsic.SHIFT_LEFT:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('lsl x0, x1, x0')
                write_level1('push x0')
            elif op.operand == Intrinsic.SWAP:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('push x0')
                write_level1('push x1')
            elif op.operand == Intrinsic.OVER:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('push x1')
                write_level1('push x0')
                write_level1('push x1')
            elif op.operand == Intrinsic.ROT:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('pop x2')
                write_level1('push x1')
                write_level1('push x0')
                write_level1('push x2')
            elif op.operand == Intrinsic.CAST_PTR:
                continue
            elif op.operand == Intrinsic.HERE:
                here_text = f'{op.token.loc.filename}:[{op.token.loc.line + 1}:{op.token.loc.col}]: '
                write_level1(f'ldr x0, ={len(here_text)}')
                write_level1('push x0')
                address = allocated_strs.get(here_text, len(strs))
                write_level1(f'adrp x1, str_{address}@PAGE')
                write_level1(f'add x1, x1, str_{address}@PAGEOFF')
                write_level1('push x1')
                if here_text not in allocated_strs:
                    allocated_strs[here_text] = len(strs)
                    strs.append(here_text)
            elif op.operand == Intrinsic.SYSCALL1:
                write_level1('pop x16')
                write_level1('pop x0')
                write_level1('svc #0')
                write_level1(f'b.cc return_ok_{i}')
                write_level1('mov x0, #-1')
                write_base(f'return_ok_{i}:')
                write_level1('push x0')
            elif op.operand == Intrinsic.SYSCALL3:
                write_level1('pop x16')
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('pop x2')
                write_level1('svc #0')
                write_level1(f'b.cc return_ok_{i}')
                write_level1('mov x0, #-1')
                write_base(f'return_ok_{i}:')
                write_level1('push x0')
            else:
                raise_error(f'Unhandled intrinsic: {op.name}',
                            op.token.loc)
        else:
            raise_error(f'Unhandled operation: {op.name}',
                        op.token.loc)
    write_level1('mov x16, #1')
    write_level1('mov x0, #0')
    write_level1('svc #0')
    write_base('.section __DATA, __data')
    write_level1('argc: .quad 0')
    write_level1('argv: .quad 0')
    for i in range(len(strs)):
        word = repr(strs[i]).strip("'")
        write_level1(f'str_{i}: .asciz "{word}"')
    write_base('.section __DATA, __bss')
    write_base('mem:')
    write_level1(f'.skip {MEM_CAPACITY}')
    out.close()


def usage_help() -> None:
    print('Usage: photon.py <SUBCOMMAND> <FLAGS> <FILENAME>')
    print('Subcommands:')
    print('     sim     Simulate the program in a macos little endian environment')
    print('     com     Compile the program to arm 64-bit assembly')
    print('         --run   Used with `com` to run immediately')


def parse_keyword(stack: List[Tuple[Token, int]], token: Token, i: int, program: List[Op]) -> NoReturn | Op:
    assert len(Keyword) == 7, 'Exhaustive handling of keywords in parse_keyword'
    if type(token.value) != Keyword:
        raise_error(f'Token value `{token.value}` must be a Keyword, but found: {type(token.value)}', token.loc)
    if token.value == Keyword.IF:
        stack.append((token, i))
        return Op(type=OpType.IF, token=token, name=token.name)
    elif token.value == Keyword.ELSE:
        if len(stack) == 0:
            raise_error(f'`else` can only be used with an `if`', token.loc)
        if_index = stack.pop()[1]
        if program[if_index].type != OpType.IF:
            if if_index:
                notify_user(f'Instead of `else` found: {program[if_index].type}', program[if_index].token.loc)
            raise_error(f'`else` can only be used with an `if`', token.loc)
        program[if_index].operand = i
        stack.append((token, i))
        return Op(type=OpType.ELSE, token=token, name=token.name)
    elif token.value == Keyword.WHILE:
        stack.append((token, i))
        return Op(type=OpType.WHILE, token=token, name=token.name)
    elif token.value == Keyword.DO:
        stack.append((token, i))
        return Op(type=OpType.DO, token=token, name=token.name)
    elif token.value == Keyword.END:
        if len(stack) == 0:
            raise_error('`end` can only be used with an `if`, `else`, `while` or `macro`',
                        token.loc)
        block_index = stack.pop()[1]
        if program[block_index].type in (OpType.IF, OpType.ELSE):
            program[block_index].operand = i
            return Op(type=OpType.END, token=token, name=token.name)
        elif program[block_index].type == OpType.DO:
            program[block_index].operand = i
            if len(stack) == 0:
                raise_error('`while` must be present before `do`', program[block_index].token.loc)
            while_index = stack.pop()[1]
            if program[while_index].type != OpType.WHILE:
                if while_index:
                    notify_user(f'Instead of `while` found: {program[while_index].type}', program[while_index].token.loc)
                raise_error('`while` must be present before `do`', program[block_index].token.loc)
            value = while_index
            return Op(type=OpType.END, token=token, name=token.name, operand=value)
        else:
            raise_error('`end` can only be used with an `if`, `else`, `while` or `macro`',
                        token.loc)
    else:
        raise_error(f'Unknown keyword token: {token.value}', token.loc)


def expand_keyword_to_tokens(token: Token, rprogram: List[Token], macros: Dict[str, Token]) -> NoReturn | None:
    assert len(Keyword) == 7, 'Exhaustive handling of keywords in compile_keyword_to_program'
    if token.value == Keyword.MACRO:
        if len(rprogram) == 0:
            raise_error('Expected name of the macro but found nothing', token.loc)
        macro_name = rprogram.pop()
        if type(macro_name.value) == Keyword:
            raise_error(f'Redefinition of keyword: `{macro_name.value.name.lower()}`', macro_name.loc)
        if macro_name.type != TokenType.WORD or type(macro_name.value) != str:
            raise_error(f'Expected macro name to be: `word`, but found: `{macro_name.name}`', macro_name.loc)
        if macro_name.value in INTRINSIC_NAMES:
            raise_error(f'Redefinition of intrinsic word: `{macro_name.value}`', macro_name.loc)
        if macro_name.value in macros:
            notify_user(f'Macro `{macro_name.value}` was defined at this location', macros[macro_name.value].loc)
            raise_error(f'Redefinition of existing macro: `{macro_name.value}`', macro_name.loc)
        if len(rprogram) == 0:
            raise_error(f'Expected `end` at the end of empty macro definition but found: `{macro_name.value}`',
                        macro_name.loc)
        macros[macro_name.value] = Token(TokenType.KEYWORD, Keyword.MACRO,
                                         loc=token.loc, name=macro_name.value, tokens=[])
        block_count = 0
        while len(rprogram) > 0:
            next_token = rprogram.pop()
            if next_token.type == TokenType.KEYWORD and next_token.value == Keyword.END:
                if block_count == 0:
                    break
                block_count -= 1
            elif next_token.type == TokenType.KEYWORD and next_token.value in (
                    Keyword.MACRO, Keyword.IF, Keyword.WHILE):
                block_count += 1
            macro_tokens = macros[macro_name.value].tokens
            assert macro_tokens is not None, 'Macro tokens not saved'
            macro_tokens.append(next_token)
        if next_token.type != TokenType.KEYWORD or next_token.value != Keyword.END:
            raise_error(f'Expected `end` at the end of macro definition but found: `{next_token.value}`',
                        next_token.loc)
    elif token.value == Keyword.INCLUDE:
        if len(rprogram) == 0:
            raise_error('Expected name of the include file but found nothing', token.loc)
        include_name = rprogram.pop()
        if include_name.type != TokenType.STR or type(include_name.value) != str:
            raise_error(f'Expected macro name to be: `string`, but found: `{include_name.name}`', include_name.loc)
        if not include_name.value.endswith('phtn'):
            raise_error(
                f'Expected include file to end with `.phtn`, but found: `{include_name.value.split(".")[-1]}`',
                include_name.loc)
        include_filepath = os.path.join('.', include_name.value)
        std_filepath = os.path.join('./std/', include_name.value)
        found = False
        for include_filepath in (include_filepath, std_filepath):
            if os.path.isfile(include_filepath):
                lexed_include = lex_file(include_filepath)
                rprogram.extend(reversed(lexed_include))
                found = True
        if not found:
            raise_error(f'Photon file with name: {include_name.value} not found', include_name.loc)
    else:
        raise_error(f'Keyword token not compilable to tokens: {token.value}', token.loc)
    return None


def parse_tokens_to_program(token_program: List[Token]) -> List[Op]:
    assert len(TokenType) == 5, "Exhaustive handling of tokens in parse_tokens_to_program."
    stack: List[Tuple[Token, int]] = []
    rprogram = list(reversed(token_program))
    program: List[Op] = []
    macros: Dict[str, Token] = {}
    i = 0
    while len(rprogram) > 0:
        token = rprogram.pop()
        if token.value in macros:
            assert type(token.value) == str, 'Compiler Error: non string macro name was saved'
            current_macro = macros[token.value]
            current_macro.expanded_count += 1
            if current_macro.expanded_count > MACRO_EXPANSION_LIMIT:
                raise_error(f'Expansion limit reached for macro: {token.value}', current_macro.loc)
            assert current_macro.tokens is not None, 'Macro tokens not saved'
            for idx in range(len(current_macro.tokens) - 1, -1, -1):
                current_macro.tokens[idx].expanded_from = token
                current_macro.tokens[idx].expanded_count = token.expanded_count + 1
                rprogram.append(current_macro.tokens[idx])
            continue
        if token.type == TokenType.KEYWORD and token.value in (Keyword.MACRO, Keyword.INCLUDE):
            expand_keyword_to_tokens(token, rprogram, macros)
        else:
            program.append(parse_token_as_op(stack, token, i, program))
            i += 1
    if len(stack) != 0:
        raise_error('Found an unclosed block', stack[-1][0].loc)
    return program


def parse_token_as_op(stack: List[Tuple[Token, int]], token: Token, i: int, program: List[Op]) -> Op | NoReturn:
    assert len(OpType) == 8, 'Exhaustive handling of built-in words'
    assert len(TokenType) == 5, 'Exhaustive handling of tokens in parser'

    if token.type == TokenType.INT:
        if type(token.value) != int:
            raise_error('Token value must be an integer', token.loc)
        return Op(type=OpType.PUSH_INT, operand=token.value, token=token, name=token.name)
    elif token.type == TokenType.CHAR:
        if type(token.value) != int:
            raise_error('Token value must be an integer', token.loc)
        return Op(type=OpType.PUSH_INT, operand=token.value, token=token, name=token.name)
    elif token.type == TokenType.STR:
        if type(token.value) != str:
            raise_error('Token value must be an string', token.loc)
        return Op(type=OpType.PUSH_STR, operand=token.value, token=token, name=token.name)
    elif token.type == TokenType.KEYWORD:
        return parse_keyword(stack, token, i, program)
    elif token.type == TokenType.WORD:
        assert type(token.value) == str, "`word` must be a string"
        if token.value not in INTRINSIC_NAMES:
            raise_error(f'Unknown intrinsic name: `{token.value}`', token.loc)
        return Op(type=OpType.INTRINSIC, operand=INTRINSIC_NAMES[token.value], token=token, name=token.value)
    else:
        raise_error(f'Unhandled token: {token}', token.loc)


def seek_until(line: str, start: int, predicate: Callable[[str], bool]) -> int:
    while start < len(line) and not predicate(line[start]):
        start += 1
    return start


def lex_word(word: str, location: Loc) -> Token:
    assert len(TokenType) == 5, "Exhaustive handling of tokens in lexer"
    if word[0] == '"':
        return Token(type=TokenType.STR,
                     value=word.strip('"').encode('utf-8').decode('unicode_escape'),
                     loc=location,
                     name='string')
    elif word[0] == "'":
        return Token(type=TokenType.CHAR,
                     value=ord(word.strip("'").encode('utf-8').decode('unicode_escape')),
                     loc=location,
                     name='char')
    else:
        try:
            return Token(type=TokenType.INT, value=int(word), loc=location, name='integer')
        except ValueError:
            if word in KEYWORD_NAMES:
                return Token(type=TokenType.KEYWORD, value=KEYWORD_NAMES[word], loc=location, name='Keyword')
            return Token(type=TokenType.WORD, value=word, loc=location, name='word')


def lex_line(line: str, file_path: str, line_number: int) -> Generator[Token, None, None]:
    col = seek_until(line, 0, lambda x: not x.isspace())
    while col < len(line):
        location = Loc(file_path, line_number, col)
        if line[col] == '"':
            col_end = seek_until(line, col + 1, lambda x: x == '"')
            if col_end >= len(line):
                raise_error('String literal was not closed', location)
            word = line[col:col_end + 1]
            col = seek_until(line, col_end + 1, lambda x: not x.isspace())
        elif line[col] == "'":
            col_end = seek_until(line, col + 1, lambda x: x == "'")
            if col_end >= len(line):
                raise_error('Char literal was not closed', location)
            word = line[col:col_end + 1]
            if len(word) > 4 or len(word.replace('\\', '')) > 3:
                raise_error('Char literal must have length 1', location)
            col = seek_until(line, col_end + 1, lambda x: not x.isspace())
        else:
            col_end = seek_until(line, col, lambda x: x.isspace())
            word = line[col:col_end]
            col = seek_until(line, col_end, lambda x: not x.isspace())
        yield lex_word(word, location)


def lex_file(file_path: str) -> List[Token]:
    program: List[Token] = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            program.extend(lex_line(line.split('//')[0], file_path, i))
    return program


if __name__ == '__main__':
    _, *argv= sys.argv
    if len(argv) < 2:
        usage_help()
        exit(1)
    subcommand, *argv = argv
    if subcommand not in ('sim', 'com'):
        usage_help()
        exit(1)
    file_path_arg = argv[0]
    program_stack = lex_file(file_path_arg)
    program_referenced = parse_tokens_to_program(program_stack)
    type_check_program(program_referenced, '-d' in argv or '--debug' in argv)
    if subcommand == 'sim':
        simulate_little_endian_macos(program_referenced, argv)
    else:
        compile_program(program_referenced)
        exit_code = subprocess.call('as -o output.o output.s', shell=True)
        if exit_code != 0:
            exit(exit_code)
        exit_code = subprocess.call(
            'ld -o output output.o', shell=True)
        if exit_code != 0:
            exit(exit_code)
        if '--run' in argv:
            args_start = argv.index('--run') + 1
            args = ''.join(argv[args_start:])
            exit(subprocess.call(f'./output {args}', shell=True))

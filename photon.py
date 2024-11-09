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
    write_base('.section __TEXT, __text')
    write_base('.global _main')
    write_base('.align 3')
    write_base('.macro push reg1:req')
    write_level1(r'str \reg1, [sp, #-16]!')
    write_base('.endmacro')
    write_base('.macro pop reg1:req')
    write_level1(r'ldr \reg1, [sp], #16')
    write_base('.endmacro')
    write_base('.macro push_ret reg1:req')
    write_level1(r'str \reg1, [x20], #16')
    write_base('.endmacro')
    write_base('.macro pop_ret reg1:req')
    write_level1(r'ldr \reg1, [x20, #-16]!')
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
    write_base('_main:')
    write_level1('adrp x2, argc@PAGE')
    write_level1('add x2, x2, argc@PAGEOFF')
    write_level1('str x0, [x2]')
    write_level1('adrp x2, argv@PAGE')
    write_level1('add x2, x2, argv@PAGEOFF')
    write_level1('str x1, [x2]')
    write_level1('adrp x20, ret_stack@PAGE')
    write_level1('add x20, x20, ret_stack@PAGEOFF')


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
    BITNOT = auto()
    SHIFT_RIGHT = auto()
    SHIFT_LEFT = auto()
    SWAP = auto()
    OVER = auto()
    ROT = auto()
    LOAD = auto()
    STORE = auto()
    LOAD2 = auto()
    STORE2 = auto()
    LOAD4 = auto()
    STORE4 = auto()
    LOAD8 = auto()
    STORE8 = auto()
    SYSCALL1 = auto()
    SYSCALL2 = auto()
    SYSCALL3 = auto()
    SYSCALL6 = auto()
    ARGC = auto()
    ARGV = auto()
    CAST_PTR = auto()
    CAST_INT = auto()
    CAST_BOOL = auto()
    HERE = auto()


class OpType(Enum):
    PUSH_INT = auto()
    PUSH_STR = auto()
    PUSH_MEM = auto()
    INTRINSIC = auto()
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    WHILE = auto()
    DO = auto()
    PROC = auto()
    CALL = auto()
    RET = auto()
    END = auto()

class Keyword(Enum):
    IF = auto()
    END = auto()
    ELSE = auto()
    ELIF = auto()
    WHILE = auto()
    DO = auto()
    MACRO = auto()
    PROC = auto()
    INCLUDE = auto()
    MEMORY = auto()


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
    expanded_from: Optional['Token'] = None
    expanded_count: int = 0


@dataclass
class Macro(Token):
    tokens: List[Token] | None = None

@dataclass
class Proc:
    addr: int
    nested_proc_count: int


@dataclass
class Op:
    type: OpType
    token: Token
    name: str
    operand: int | str | Intrinsic | None = None
    addr: int | None = None


@dataclass
class Program:
    ops: list[Op]
    memory_capacity: int
    proc_ret_capacity: int


KEYWORD_NAMES = {
    'if': Keyword.IF,
    'end': Keyword.END,
    'elif': Keyword.ELIF,
    'else': Keyword.ELSE,
    'while': Keyword.WHILE,
    'do': Keyword.DO,
    'macro': Keyword.MACRO,
    'include': Keyword.INCLUDE,
    'memory': Keyword.MEMORY,
    'proc': Keyword.PROC,
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
    'bnot': Intrinsic.BITNOT,
    '>>': Intrinsic.SHIFT_RIGHT,
    '<<': Intrinsic.SHIFT_LEFT,
    'swap': Intrinsic.SWAP,
    'over': Intrinsic.OVER,
    'rot': Intrinsic.ROT,
    '>=': Intrinsic.GTE,
    '<=': Intrinsic.LTE,
    '!=': Intrinsic.NE,
    '!': Intrinsic.STORE,
    '@': Intrinsic.LOAD,
    '!2': Intrinsic.STORE2,
    '@2': Intrinsic.LOAD2,
    '!4': Intrinsic.STORE4,
    '@4': Intrinsic.LOAD4,
    '!8': Intrinsic.STORE8,
    '@8': Intrinsic.LOAD8,
    'syscall1': Intrinsic.SYSCALL1,
    'syscall2': Intrinsic.SYSCALL2,
    'syscall3': Intrinsic.SYSCALL3,
    'syscall6': Intrinsic.SYSCALL6,
    'argc': Intrinsic.ARGC,
    'argv': Intrinsic.ARGV,
    '->ptr': Intrinsic.CAST_PTR,
    '->int': Intrinsic.CAST_INT,
    '->bool': Intrinsic.CAST_BOOL,
    'here': Intrinsic.HERE,
}

assert len(INTRINSIC_NAMES) == len(Intrinsic), 'Exhaustive handling of intrinsics'

MACRO_EXPANSION_LIMIT = 100_000
MACRO_TRACEBACK_LIMIT = 10
NULL_POINTER_PADDING = 1  # padding to make 0 an invalid address
ARG_PTR_CAPACITY = 640 + NULL_POINTER_PADDING
STR_CAPACITY = 640_000 + ARG_PTR_CAPACITY
FDS: List[BinaryIO] = [sys.stdin.buffer, sys.stdout.buffer, sys.stderr.buffer]

DataTypeStack = List[Tuple[DataType, Token]]
MemAddr = int


def make_log_message(message: str, loc: Loc) -> str:
    return f'{loc.filename}:[{loc.line + 1}:{loc.col}] {message}'


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
            location = expand_place.expanded_from.loc
            if location.filename[:2] != './':
                location.filename = './' + location.filename
            notify_user(f'Operation expanded from macro: {expand_place.expanded_from.value}',
                        loc=location)
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


def type_check_program(program: Program, debug: bool = False) -> None:
    stack: DataTypeStack = []
    block_stack: List[Tuple[DataTypeStack, Op]] = []  # convert stack to tuple keeping only DataType, hash and store
    for op in program.ops:
        assert len(OpType) == 10, 'Exhaustive handling of operations in type check'
        if op.type == OpType.PUSH_INT:
            assert type(op.operand) == int, 'Value for `PUSH_INT` must be `int`'
            stack.append((DataType.INT, op.token))
        elif op.type == OpType.PUSH_STR:
            assert type(op.operand) == str, 'Value for `PUSH_STR` must be `str`'
            if op.operand[-1] != '\0':
                stack.append((DataType.INT, op.token))
            stack.append((DataType.PTR, op.token))
        elif op.type == OpType.PUSH_MEM:
            stack.append((DataType.PTR, op.token))
        elif op.type == OpType.IF:
            block_stack.append((stack.copy(), op))
        elif op.type == OpType.ELIF:
            before_do_stack, do_op = block_stack.pop()
            assert do_op.type == OpType.DO, '[BUG] elif without do'
            if_elif_stack, if_elif_block = block_stack.pop()
            assert if_elif_block.type == OpType.ELIF or if_elif_block.type == OpType.IF, '[BUG] invalid operation before do-elif'
            if if_elif_block.type == OpType.ELIF:
                expected_stack_before_block = list(map(lambda x: x[0], if_elif_stack))
                current_stack = list(map(lambda x: x[0], stack))
                if current_stack != expected_stack_before_block:
                    notify_user(f'Expected Stack Types: {expected_stack_before_block}', op.token.loc)
                    notify_user(f'Actual Stack Types: {current_stack}', op.token.loc)
                    raise_error('All branches of an if-elif block must produce the same stack types', op.token)
            block_stack.append((stack.copy(), op))
            stack = before_do_stack
        elif op.type == OpType.ELSE:
            before_do_stack, do_op = block_stack.pop()
            assert do_op.type == OpType.DO, '[BUG] else without do'
            block_stack.append((stack.copy(), op))
            stack = before_do_stack
        elif op.type == OpType.END:
            stack_before_block, block = block_stack.pop()
            expected_stack_before_block = list(map(lambda x: x[0], stack_before_block))
            current_stack = list(map(lambda x: x[0], stack))
            if block.type == OpType.ELSE:
                if current_stack != expected_stack_before_block:
                    notify_user(f'Expected Stack Types: {expected_stack_before_block}', op.token.loc)
                    notify_user(f'Actual Stack Types: {current_stack}', op.token.loc)
                    raise_error('Both branches of an if-else block must produce the same stack types', op.token)
                _, if_block = block_stack.pop()
                assert if_block.type == OpType.IF or if_block.type == OpType.ELIF, '[BUG] No `if` or `elif` before `do-else`'
            elif block.type == OpType.DO:
                stack_before, before_do_op = block_stack.pop()
                assert before_do_op.type in (
                    OpType.WHILE, OpType.IF, OpType.ELIF), '[BUG] No `while`, `if` or `elif` before `do`'
                expected_stack = list(map(lambda x: x[0], stack_before))
                if before_do_op.type == OpType.WHILE and current_stack != expected_stack:
                    notify_user(f'Expected Stack Types: {expected_stack}', op.token.loc)
                    notify_user(f'Actual Stack Types: {current_stack}', op.token.loc)
                    raise_error('Stack types cannot be altered inside a while-do body', op.token)
                if before_do_op.type == OpType.IF and current_stack != expected_stack:
                    notify_user(f'Expected Stack Types: {expected_stack}', op.token.loc)
                    notify_user(f'Actual Stack Types: {current_stack}', op.token.loc)
                    raise_error('Stack types cannot be altered after an else-less if block', op.token)
                if before_do_op.type == OpType.ELIF and current_stack != expected_stack:
                    notify_user(f'Expected Stack Types: {expected_stack}', op.token.loc)
                    notify_user(f'Actual Stack Types: {current_stack}', op.token.loc)
                    raise_error('All branches of an if-elif block must produce the same stack types', op.token)
                if before_do_op.type == OpType.WHILE or before_do_op.type == OpType.IF:
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
            if block_stack[-1][1].type == OpType.IF:
                stack_before_if, if_block = block_stack[-1]
                expected_stack_before_block = list(map(lambda x: x[0], stack_before_if))
                current_stack = list(map(lambda x: x[0], stack))
                if expected_stack_before_block != current_stack:
                    notify_user(f'Expected Stack Types: {expected_stack_before_block}', op.token.loc)
                    notify_user(f'Actual Stack Types: {current_stack}', op.token.loc)
                    raise_error('Stack types cannot be altered in an if-do condition', op.token)
            block_stack.append((stack.copy(), op))
        elif op.type == OpType.INTRINSIC:
            assert len(Intrinsic) == 39, 'Exhaustive handling of intrinsics in type check'
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
                if a_type != DataType.PTR or (b_type != DataType.INT and b_type != DataType.PTR):
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.LOAD2:
                ensure_argument_count(len(stack), op, 1)
                a_type, a_loc = stack.pop()
                if a_type != DataType.PTR:
                    if debug:
                        notify_argument_origin(a_loc, order=1)
                    raise_error(f'Invalid argument types for `{op.name}`: {a_type.name}', op.token)
                stack.append((DataType.INT, op.token))
            elif op.operand == Intrinsic.STORE2:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type != DataType.PTR or (b_type != DataType.INT and b_type != DataType.PTR):
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.LOAD4:
                ensure_argument_count(len(stack), op, 1)
                a_type, a_loc = stack.pop()
                if a_type != DataType.PTR:
                    if debug:
                        notify_argument_origin(a_loc, order=1)
                    raise_error(f'Invalid argument types for `{op.name}`: {a_type.name}', op.token)
                stack.append((DataType.INT, op.token))
            elif op.operand == Intrinsic.STORE4:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type != DataType.PTR or (b_type != DataType.INT and b_type != DataType.PTR):
                    if debug:
                        notify_argument_origin(b_loc, order=1)
                        notify_argument_origin(a_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(b_type.name, a_type.name)}', op.token)
            elif op.operand == Intrinsic.LOAD8:
                ensure_argument_count(len(stack), op, 1)
                a_type, a_loc = stack.pop()
                if a_type != DataType.PTR:
                    if debug:
                        notify_argument_origin(a_loc, order=1)
                    raise_error(f'Invalid argument types for `{op.name}`: {a_type.name}', op.token)
                stack.append((DataType.INT, op.token))
            elif op.operand == Intrinsic.STORE8:
                ensure_argument_count(len(stack), op, 2)
                a_type, a_loc = stack.pop()
                b_type, b_loc = stack.pop()
                if a_type != DataType.PTR or (b_type != DataType.INT and b_type != DataType.PTR):
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
            elif op.operand == Intrinsic.BITNOT:
                ensure_argument_count(len(stack), op, 1)
                a_type, a_loc = stack.pop()
                if a_type == DataType.INT:
                    stack.append((DataType.INT, op.token))
                else:
                    if debug:
                        notify_argument_origin(a_loc, order=1)
                    raise_error(f'Invalid argument types for `{op.name}`: {a_type.name}', op.token)
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
                if a_type != DataType.INT and a_type != DataType.PTR:
                    if debug:
                        notify_argument_origin(a_loc, order=1)
                    raise_error(f'Invalid argument types for `{op.name}`: {a_type.name}', op.token)
                stack.append((DataType.PTR, op.token))
            elif op.operand == Intrinsic.CAST_INT:
                ensure_argument_count(len(stack), op, 1)
                a_type, a_loc = stack.pop()
                if a_type != DataType.BOOL and a_type != DataType.INT:
                    if debug:
                        notify_argument_origin(a_loc, order=1)
                    raise_error(f'Invalid argument types for `{op.name}`: {a_type.name}', op.token)
                stack.append((DataType.INT, op.token))
            elif op.operand == Intrinsic.CAST_BOOL:
                ensure_argument_count(len(stack), op, 1)
                a_type, a_loc = stack.pop()
                if a_type != DataType.INT and a_type != DataType.BOOL:
                    if debug:
                        notify_argument_origin(a_loc, order=1)
                    raise_error(f'Invalid argument types for `{op.name}`: {a_type.name}', op.token)
                stack.append((DataType.BOOL, op.token))
            elif op.operand == Intrinsic.HERE:
                stack.append((DataType.INT, op.token))
                stack.append((DataType.PTR, op.token))
            elif op.operand == Intrinsic.SYSCALL1:
                ensure_argument_count(len(stack), op, 2)
                syscall_type, syscall_loc = stack.pop()
                arg1_type, arg1_loc = stack.pop()
                if syscall_type != DataType.INT:
                    if debug:
                        notify_argument_origin(arg1_loc, order=1)
                        notify_argument_origin(syscall_loc, order=2)
                    raise_error(f'Invalid argument types for `{op.name}`: {(arg1_type.name, syscall_type.name)}',
                                op.token)
                stack.append((DataType.INT, op.token))
            elif op.operand == Intrinsic.SYSCALL2:
                ensure_argument_count(len(stack), op, 3)
                syscall_type, syscall_loc = stack.pop()
                arg2_type, arg2_loc = stack.pop()
                arg1_type, arg1_loc = stack.pop()
                if syscall_type != DataType.INT:
                    if debug:
                        notify_argument_origin(arg1_loc, order=1)
                        notify_argument_origin(arg2_loc, order=2)
                        notify_argument_origin(syscall_loc, order=3)
                    raise_error(
                        f'Invalid argument types for `{op.name}`: '
                        f'{(arg1_type.name, arg2_type.name, syscall_type.name)}',
                        op.token)
                stack.append((DataType.INT, op.token))
            elif op.operand == Intrinsic.SYSCALL3:
                ensure_argument_count(len(stack), op, 4)
                syscall_type, syscall_loc = stack.pop()
                arg3_type, arg3_loc = stack.pop()
                arg2_type, arg2_loc = stack.pop()
                arg1_type, arg1_loc = stack.pop()
                if syscall_type != DataType.INT:
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
            elif op.operand == Intrinsic.SYSCALL6:
                ensure_argument_count(len(stack), op, 7)
                syscall_type, syscall_loc = stack.pop()
                arg6_type, arg6_loc = stack.pop()
                arg5_type, arg5_loc = stack.pop()
                arg4_type, arg4_loc = stack.pop()
                arg3_type, arg3_loc = stack.pop()
                arg2_type, arg2_loc = stack.pop()
                arg1_type, arg1_loc = stack.pop()
                if syscall_type != DataType.INT:
                    if debug:
                        notify_argument_origin(arg1_loc, order=1)
                        notify_argument_origin(arg2_loc, order=2)
                        notify_argument_origin(arg3_loc, order=3)
                        notify_argument_origin(arg4_loc, order=4)
                        notify_argument_origin(arg5_loc, order=5)
                        notify_argument_origin(arg6_loc, order=6)
                        notify_argument_origin(syscall_loc, order=7)
                    raise_error(
                        f'Invalid argument types for `{op.name}`: '
                        f'{arg1_type.name, arg2_type.name, arg3_type.name, arg4_type.name}'
                        f'{arg5_type.name, arg6_type.name, syscall_type.name}',
                        op.token)
                stack.append((DataType.INT, op.token))
            else:
                raise_error(f'Unhandled intrinsic: {op.name}',
                            op.token.loc)
        else:
            raise_error(f'Unhandled op: {op.name}',
                        op.token.loc)
    assert len(block_stack) == 0, '[BUG] Block Stack Not Empty'
    if len(stack) != 0:
        current_stack = list(map(lambda x: x[0], stack))
        raise_error(f'Unhandled data on the stack: {current_stack}', stack[-1][1])


def simulate_little_endian_macos(program: Program, input_arguments: List[str]) -> None:
    assert len(OpType) == 13, 'Exhaustive handling of operators in simulation'
    stack: List[int] = []
    return_stack: List[int] = []
    i = 0
    mem = bytearray(STR_CAPACITY + program.memory_capacity)
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
    while i < len(program.ops):
        op = program.ops[i]
        if op.type == OpType.PUSH_INT:
            assert type(op.operand) == int, 'Value for `PUSH_INT` must be `int`'
            stack.append(op.operand)
        elif op.type == OpType.PUSH_STR:
            assert type(op.operand) == str, 'Value for `PUSH_STR` must be `str`'
            bs = bytes(op.operand, 'utf-8')
            n = len(bs)
            if op.operand[-1] != '\0':
                stack.append(n)
            if op.operand not in allocated_strs:
                allocated_strs[op.operand] = str_size
                mem[str_size:str_size + n] = bs
                mem[str_size + n] = 0
                str_size += n
                if op.operand[-1] != '\0':
                    str_size += 1
                if str_size > STR_CAPACITY:
                    raise_error('String buffer overflow', op.token.loc)
            stack.append(allocated_strs[op.operand])
        elif op.type == OpType.PUSH_MEM:
            assert type(op.operand) == int, 'Operand for `PUSH_MEM` must be `int`'
            stack.append(STR_CAPACITY + op.operand)
        elif op.type == OpType.IF:
            i += 1
            continue
        elif op.type == OpType.ELIF:
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
        elif op.type == OpType.PROC:
            i = op.operand
        elif op.type == OpType.CALL:
            return_stack.append(i)
            i = op.operand
        elif op.type == OpType.RET:
            i = return_stack.pop()
        elif op.type == OpType.INTRINSIC:
            assert len(Intrinsic) == 39, 'Exhaustive handling of intrinsics in simulation'
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
            elif op.operand == Intrinsic.ARGC:
                stack.append(argc)
            elif op.operand == Intrinsic.ARGV:
                stack.append(argv_start)
            elif op.operand == Intrinsic.LOAD:
                address = stack.pop()
                assert type(address) == int, 'Arguments for `,` must be `int`'
                stack.append(mem[address])
            elif op.operand == Intrinsic.STORE:
                address = stack.pop()
                value = stack.pop()
                assert type(value) == type(address) == int, 'Arguments for `.` must be `int`'
                mem[address] = value & 0xFF
            elif op.operand == Intrinsic.LOAD2:
                addr = stack.pop()
                stack.append(int.from_bytes(mem[addr:addr + 2], byteorder="little"))
            elif op.operand == Intrinsic.STORE2:
                store_addr16 = stack.pop()
                store_value16 = stack.pop()
                mem[store_addr16:store_addr16 + 2] = store_value16.to_bytes(length=2, byteorder="little")
            elif op.operand == Intrinsic.LOAD4:
                addr = stack.pop()
                stack.append(int.from_bytes(mem[addr:addr + 4], byteorder="little"))
            elif op.operand == Intrinsic.STORE4:
                store_addr32 = stack.pop()
                store_value32 = stack.pop()
                mem[store_addr32:store_addr32 + 4] = store_value32.to_bytes(length=4, byteorder="little")
            elif op.operand == Intrinsic.LOAD8:
                addr = stack.pop()
                stack.append(int.from_bytes(mem[addr:addr + 8], byteorder="little"))
            elif op.operand == Intrinsic.STORE8:
                store_addr64 = stack.pop()
                store_value64 = stack.pop()
                mem[store_addr64:store_addr64 + 8] = store_value64.to_bytes(length=8, byteorder="little")
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
            elif op.operand == Intrinsic.BITNOT:
                a = stack.pop()
                assert type(a) == int, 'Argument for `bnot` must be `int`'
                stack.append(~a & 0xFFFFFFFFFFFFFFFF)
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
            elif op.operand == Intrinsic.CAST_INT:
                i += 1
                continue
            elif op.operand == Intrinsic.CAST_BOOL:
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
            elif op.operand == Intrinsic.SYSCALL2:
                syscall_number = stack.pop()
                arg1 = stack.pop()
                arg2 = stack.pop()
                assert type(arg1) == type(arg2) == int, 'Arguments for `syscall2` must be `int`'
                if syscall_number == 5:
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
                    raise_error('Unsupported syscall number', op.token)
            elif op.operand == Intrinsic.SYSCALL3:
                syscall_number = stack.pop()
                arg1 = stack.pop()
                arg2 = stack.pop()
                arg3 = stack.pop()
                assert type(arg1) == type(arg2) == type(arg3) == int, 'Arguments for `syscall3` must be `int`'
                if syscall_number == 3:
                    data = FDS[arg1].readline(arg3)
                    mem[arg2:arg2 + len(data)] = data
                    stack.append(len(data))
                elif syscall_number == 4:
                    FDS[arg1].write(mem[arg2:arg2 + arg3])
                    FDS[arg1].flush()
                    stack.append(arg3)
                else:
                    raise_error(f'Unknown syscall number: {syscall_number}', op.token.loc)
            else:
                raise_error(f'Unhandled intrinsic: {op.name}',
                            op.token.loc)
        else:
            raise_error(f'Unhandled operation: {op.name}',
                        op.token.loc)
        i += 1


def compile_program(program: Program) -> None:
    assert len(OpType) == 13, 'Exhaustive handling of operators in compilation'
    out = open('output.s', 'w')
    write_base = write_indent(out, 0)
    write_level1 = write_indent(out, 1)
    asm_setup(write_base, write_level1)
    strs: List[str] = []
    allocated_strs: Dict[str, int] = {}
    for i in range(len(program.ops)):
        op = program.ops[i]
        if op.type == OpType.PUSH_INT:
            assert type(op.operand) == int, 'Operation value must be an `int` for PUSH_INT'
            write_level1(f'ldr x0, ={op.operand}')
            write_level1('push x0')
        elif op.type == OpType.PUSH_STR:
            assert type(op.operand) == str, 'Operation value must be a `str` for PUSH_STR'
            if op.operand[-1] != '\0':
                write_level1(f'ldr x0, ={len(op.operand)}')
                write_level1('push x0')
            address = allocated_strs.get(op.operand, len(strs))
            write_level1(f'adrp x1, str_{address}@PAGE')
            write_level1(f'add x1, x1, str_{address}@PAGEOFF')
            write_level1('push x1')
            if op.operand not in allocated_strs:
                allocated_strs[op.operand] = len(strs)
                strs.append(op.operand)
        elif op.type == OpType.PUSH_MEM:
            write_level1('adrp x0, mem@PAGE')
            write_level1('add x0, x0, mem@PAGEOFF')
            write_level1(f'ldr x1, ={op.operand}')
            write_level1('add x0, x0, x1')
            write_level1('push x0')
        elif op.type == OpType.DO:
            write_level1('pop x0')
            write_level1('tst x0, x0')
            assert op.operand is not None, 'No address to jump'
            write_level1(f'b.eq end_{op.operand}')
        elif op.type == OpType.ELSE:
            write_level1(f'b end_{op.operand}')
            write_base(f'end_{i}:')
        elif op.type == OpType.ELIF:
            write_base(f'end_{i - 1}:')
            write_level1(f'b end_{op.operand}')
            write_base(f'end_{i}:')
        elif op.type == OpType.END:
            if op.operand is not None:
                write_level1(f'b while_{op.operand}')
            write_base(f'end_{i}:')
        elif op.type == OpType.WHILE:
            write_base(f'while_{i}:')
        elif op.type == OpType.IF:
            write_level1(';; -- if --')
        elif op.type == OpType.PROC:
            write_level1(f'b ret_{op.operand}')
            write_base(f'proc_{i}:')
            write_level1('push_ret lr')
        elif op.type == OpType.CALL:
            write_level1(f'bl proc_{op.operand}')
        elif op.type == OpType.RET:
            write_level1('pop_ret lr')
            write_level1(f'ret')
            write_base(f'ret_{i}:')
        elif op.type == OpType.INTRINSIC:
            assert len(Intrinsic) == 39, 'Exhaustive handling of intrinsics in simulation'
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
                write_level1('pop x1')
                write_level1('pop x0')
                write_level1('strb w0, [x1]')
            elif op.operand == Intrinsic.LOAD:
                write_level1('pop x0')
                write_level1('ldrb w1, [x0]')
                write_level1('push x1')
            elif op.operand == Intrinsic.STORE2:
                write_level1('pop x1')
                write_level1('pop x0')
                write_level1('strh w0, [x1]')
            elif op.operand == Intrinsic.LOAD2:
                write_level1('pop x0')
                write_level1('ldrh w1, [x0]')
                write_level1('push x1')
            elif op.operand == Intrinsic.STORE4:
                write_level1('pop x1')
                write_level1('pop x0')
                write_level1('str w0, [x1]')
            elif op.operand == Intrinsic.LOAD4:
                write_level1('pop x0')
                write_level1('ldr w1, [x0]')
                write_level1('push x1')
            elif op.operand == Intrinsic.STORE8:
                write_level1('pop x1')
                write_level1('pop x0')
                write_level1('str x0, [x1]')
            elif op.operand == Intrinsic.LOAD8:
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
            elif op.operand == Intrinsic.BITNOT:
                write_level1('pop x0')
                write_level1('mvn x0, x0')
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
            elif op.operand == Intrinsic.CAST_INT:
                continue
            elif op.operand == Intrinsic.CAST_BOOL:
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
            elif op.operand == Intrinsic.SYSCALL2:
                write_level1('pop x16')
                write_level1('pop x0')
                write_level1('pop x1')
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
            elif op.operand == Intrinsic.SYSCALL6:
                write_level1('pop x16')
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('pop x2')
                write_level1('pop x3')
                write_level1('pop x4')
                write_level1('pop x5')
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
    write_level1(f'.skip {program.memory_capacity}')
    write_base('ret_stack:')
    write_level1(f'.skip {(program.proc_ret_capacity + 2) * 16}')
    out.close()


def usage_help() -> None:
    print('Usage: photon.py <SUBCOMMAND> <FLAGS> <FILENAME>')
    print('Subcommands:')
    print('     sim     Simulate the program in a macos little endian environment')
    print('     flow    Generate a control flow graph for the given program')
    print('     com     Compile the program to arm 64-bit assembly')
    print('         --run   Used with `com` to run immediately')


class Parsing:
    def __init__(self, token_program: List[Token]) -> None:
        self.current_proc: str | None = None
        self.program = Program([], 0, 0)
        self.stack: List[Tuple[Token, int]] = []
        self.rprogram = list(reversed(token_program))
        self.macros: Dict[str, Macro] = {}
        self.memories: Dict[str, MemAddr] = {}
        self.procs: Dict[str, Proc] = {}
        self.index = 0
        self.current_proc_ret_cap = 0

    def parse_tokens_to_program(self) -> Program:
        assert len(OpType) == 13, "Exhaustive handling of op types in parse_tokens_to_program."
        assert len(Keyword) == 10, "Exhaustive handling of keywords in parse_tokens_to_program."
        while len(self.rprogram) > 0:
            token = self.rprogram.pop()
            if token.value in self.macros:
                assert type(token.value) == str, 'Compiler Error: non string macro name was saved'
                current_macro = self.macros[token.value]
                current_macro.expanded_count += 1
                if current_macro.expanded_count > MACRO_EXPANSION_LIMIT:
                    raise_error(f'Expansion limit reached for macro: {token.value}', current_macro.loc)
                assert current_macro.tokens is not None, 'Macro tokens not saved'
                for idx in range(len(current_macro.tokens) - 1, -1, -1):
                    current_macro.tokens[idx].expanded_from = token
                    current_macro.tokens[idx].expanded_count = token.expanded_count + 1
                    self.rprogram.append(current_macro.tokens[idx])
            elif token.value in self.memories:
                assert type(token.value) == str, 'Compiler Error: non string memory name was saved'
                self.program.ops.append(
                    Op(type=OpType.PUSH_MEM, token=token, operand=self.memories[token.value], name=token.value))
                self.index += 1
            elif token.value in self.procs:
                assert type(token.value) == str, 'Compiler Error: non string process name was saved'
                self.program.ops.append(Op(type=OpType.CALL, token=token,
                                           operand=self.procs[token.value].addr, name=token.value))
                if self.current_proc is not None:
                    self.procs[self.current_proc].nested_proc_count = max(self.procs[self.current_proc].nested_proc_count,
                                                                          self.procs[token.value].nested_proc_count + 1)
                self.index += 1
            elif token.type == TokenType.KEYWORD and token.value in (Keyword.MACRO, Keyword.INCLUDE):
                self.expand_keyword_to_tokens(token)
            elif token.type == TokenType.KEYWORD and token.value == Keyword.MEMORY:
                memory_name, memory_size = self.evaluate_memory_definition(token)
                self.memories[memory_name] = self.program.memory_capacity
                self.program.memory_capacity += memory_size
            else:
                self.program.ops.append(self.parse_token_as_op(token))
                self.index += 1
        if len(self.stack) != 0:
            raise_error('Found an unclosed block', self.stack[-1][0].loc)
        return self.program

    def parse_keyword(self, token: Token) -> NoReturn | Op:
        assert len(Keyword) == 10, 'Exhaustive handling of keywords in parse_keyword'
        if type(token.value) != Keyword:
            raise_error(f'Token value `{token.value}` must be a Keyword, but found: {type(token.value)}', token.loc)
        if token.value == Keyword.IF:
            self.stack.append((token, self.index))
            return Op(type=OpType.IF, token=token, name=token.value.name)
        elif token.value == Keyword.ELIF:
            if len(self.stack) < 2:
                raise_error(f'`elif` can only be used with an `if-do` or `elif-do`', token.loc)
            _, do_index = self.stack.pop()
            if self.program.ops[do_index].type != OpType.DO:
                if do_index:
                    notify_user(f'Instead of `do` found: {self.program.ops[do_index].type}',
                                self.program.ops[do_index].token.loc)
                raise_error(f'`elif` can only be used with an `if-do` or `elif-do`', token.loc)
            _, if_index = self.stack.pop()
            if self.program.ops[if_index].type != OpType.IF and self.program.ops[if_index].type != OpType.ELIF:
                notify_user(f'Instead of `if` or `elif` found: {self.program.ops[if_index].type}',
                            self.program.ops[if_index].token.loc)
                raise_error('`if` or `elif` must be present before `do`', self.program.ops[if_index].token.loc)
            if self.program.ops[if_index].type == OpType.ELIF:
                self.program.ops[if_index].operand = self.index - 1
            self.program.ops[do_index].operand = self.index
            self.stack.append((token, self.index))
            return Op(type=OpType.ELIF, token=token, name=token.value.name)
        elif token.value == Keyword.ELSE:
            if len(self.stack) < 2:
                raise_error(f'`else` can only be used with an `if-do` or `elif-do`', token.loc)
            _, do_index = self.stack.pop()
            if self.program.ops[do_index].type != OpType.DO and self.program.ops[do_index].type != OpType.ELIF:
                if do_index:
                    notify_user(f'Instead of `do` found: {self.program.ops[do_index].type}',
                                self.program.ops[do_index].token.loc)
                raise_error(f'`else` can only be used with an `if-do` or `elif-do`', token.loc)
            self.program.ops[do_index].operand = self.index
            self.stack.append((token, self.index))
            return Op(type=OpType.ELSE, token=token, name=token.value.name)
        elif token.value == Keyword.WHILE:
            self.stack.append((token, self.index))
            return Op(type=OpType.WHILE, token=token, name=token.value.name)
        elif token.value == Keyword.DO:
            self.stack.append((token, self.index))
            return Op(type=OpType.DO, token=token, name=token.value.name)
        elif token.value == Keyword.PROC:
            if self.current_proc is not None:
                raise_error('Nested `proc` blocks are not allowed', token.loc)
            if len(self.rprogram) == 0:
                raise_error('Expected name of the procedure but found nothing', token.loc)
            proc_name = self.rprogram.pop()
            self.check_block_name_validity(proc_name)
            if len(self.rprogram) == 0:
                raise_error(f'Expected `end` at the end of empty procedure definition but found: `{proc_name.value}`',
                            proc_name.loc)
            assert type(proc_name.value) == str, 'Procedure name value must be a string'
            self.current_proc = proc_name.value
            self.procs[proc_name.value] = Proc(self.index, 0)
            self.stack.append((token, self.index))
            return Op(type=OpType.PROC, token=token, name=token.value.name)
        elif token.value == Keyword.END:
            if len(self.stack) == 0:
                raise_error('`end` can only be used with a `if-do`, `if-do-else`, `while-do`, `proc` or `macro`',
                            token.loc)
            block, block_index = self.stack.pop()
            if self.program.ops[block_index].type == OpType.ELSE:
                self.program.ops[block_index].operand = self.index
                if len(self.stack) == 0:
                    raise_error('`if-do` must be present before `else`', self.program.ops[block_index].token.loc)
                _, if_index = self.stack.pop()
                if self.program.ops[if_index].type != OpType.IF and self.program.ops[if_index].type != OpType.ELIF:
                    notify_user(f'Instead of `if` or `elif` found: {self.program.ops[if_index].type}',
                                self.program.ops[if_index].token.loc)
                    raise_error('`if` or `elif` must be present before `do`', self.program.ops[if_index].token.loc)
                if self.program.ops[if_index].type == OpType.ELIF:
                    self.program.ops[if_index].operand = self.index
                return Op(type=OpType.END, token=token, name=token.value.name)
            elif self.program.ops[block_index].type == OpType.DO:
                self.program.ops[block_index].operand = self.index
                if len(self.stack) == 0:
                    raise_error('`while`, `if` or `elif` must be present before `do`',
                                self.program.ops[block_index].token.loc)
                _, before_do_index = self.stack.pop()
                if self.program.ops[before_do_index].type == OpType.WHILE:
                    return Op(type=OpType.END, token=token, name=token.value.name, operand=before_do_index)
                elif self.program.ops[before_do_index].type == OpType.IF:
                    return Op(type=OpType.END, token=token, name=token.value.name)
                elif self.program.ops[before_do_index].type == OpType.ELIF:
                    self.program.ops[before_do_index].operand = self.index
                    return Op(type=OpType.END, token=token, name=token.value.name)
                else:
                    notify_user(f'Instead of `while`, `if` or `elif` found: {self.program.ops[before_do_index].type}',
                                self.program.ops[before_do_index].token.loc)
                    raise_error('`while`, `if` or `elif` must be present before `do`',
                                self.program.ops[block_index].token.loc)
            elif self.program.ops[block_index].type == OpType.PROC:
                self.program.ops[block_index].operand = self.index
                self.program.proc_ret_capacity = max(self.program.proc_ret_capacity,
                                                     self.procs[self.current_proc].nested_proc_count)
                self.current_proc = None
                return Op(type=OpType.RET, token=token, name=token.value.name)
            else:
                notify_user(f'Instead of `else` or `do` found: {self.program.ops[block_index].type}',
                            self.program.ops[block_index].token.loc)
                raise_error('`end` can only be used with an `if-do`, `if-do-else`, `while-do`, `proc` or `macro`',
                            token.loc)
        else:
            raise_error(f'Unknown keyword token: {token.value}', token.loc)

    def expand_keyword_to_tokens(self, token: Token) -> NoReturn | None:
        assert len(Keyword) == 10, 'Exhaustive handling of keywords in compile_keyword_to_program'
        if token.value == Keyword.MACRO:
            if len(self.rprogram) == 0:
                raise_error('Expected name of the macro but found nothing', token.loc)
            macro_name = self.rprogram.pop()
            self.check_block_name_validity(macro_name)
            if len(self.rprogram) == 0:
                raise_error(f'Expected `end` at the end of empty macro definition but found: `{macro_name.value}`',
                            macro_name.loc)
            assert type(macro_name.value) == str, 'Macro name value must be a string'
            self.macros[macro_name.value] = Macro(TokenType.KEYWORD, Keyword.MACRO,
                                                  loc=token.loc, name=macro_name.value, tokens=[])
            block_count = 0
            while len(self.rprogram) > 0:
                next_token = self.rprogram.pop()
                assert len(Keyword) == 10, 'Exhaustive handling of keywords in macro expansion'
                if next_token.type == TokenType.KEYWORD and next_token.value == Keyword.END:
                    if block_count == 0:
                        break
                    block_count -= 1
                elif next_token.type == TokenType.KEYWORD and next_token.value in (
                        Keyword.MACRO, Keyword.IF, Keyword.WHILE, Keyword.MEMORY, Keyword.PROC):
                    block_count += 1
                macro_tokens = self.macros[macro_name.value].tokens
                assert macro_tokens is not None, 'Macro tokens not saved'
                macro_tokens.append(next_token)
            if next_token.type != TokenType.KEYWORD or next_token.value != Keyword.END:
                raise_error(f'Expected `end` at the end of macro definition but found: `{next_token.value}`',
                            next_token.loc)
        # TODO: Double include of same file leads to redefinition of macro
        elif token.value == Keyword.INCLUDE:
            if len(self.rprogram) == 0:
                raise_error('Expected name of the include file but found nothing', token.loc)
            include_name = self.rprogram.pop()
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
                    self.rprogram.extend(reversed(lexed_include))
                    found = True
            if not found:
                raise_error(f'Photon file with name: {include_name.value} not found', include_name.loc)
        else:
            raise_error(f'Keyword token not compilable to tokens: {token.value}', token.loc)
        return None

    def evaluate_memory_definition(self, token: Token) -> Tuple[str, int]:
        if len(self.rprogram) == 0:
            raise_error('Expected name of the memory but found nothing', token.loc)
        memory_name = self.rprogram.pop()
        self.check_block_name_validity(memory_name)
        if len(self.rprogram) == 0:
            raise_error(f'Expected `end` at the end of empty memory definition but found: `{memory_name.value}`',
                        memory_name.loc)
        memory_size_stack: List[int] = []
        while len(self.rprogram) > 0:
            token = self.rprogram.pop()
            if token.type == TokenType.KEYWORD and token.value == Keyword.END:
                break
            elif token.type == TokenType.INT:
                assert type(token.value) == int, 'Token value must be an integer'
                memory_size_stack.append(token.value)
            elif token.type == TokenType.WORD and type(token.value) == str:
                if INTRINSIC_NAMES.get(token.value, '') == Intrinsic.ADD:
                    # TODO: Check for memory_size_stack underflow
                    a = memory_size_stack.pop()
                    b = memory_size_stack.pop()
                    memory_size_stack.append(a + b)
                elif INTRINSIC_NAMES.get(token.value, '') == Intrinsic.MUL:
                    a = memory_size_stack.pop()
                    b = memory_size_stack.pop()
                    memory_size_stack.append(a * b)
                elif token.value in self.macros:
                    current_macro = self.macros[token.value]
                    assert current_macro.tokens is not None, 'Macro tokens not saved'
                    for idx in range(len(current_macro.tokens) - 1, -1, -1):
                        current_macro.tokens[idx].expanded_from = token
                        current_macro.tokens[idx].expanded_count = token.expanded_count + 1
                        self.rprogram.append(current_macro.tokens[idx])
                else:
                    raise_error(f'Unsupported token in memory definition: {token.value}', token.loc)
            else:
                raise_error(f'Unsupported token in memory definition: {token.value}', token.loc)
        if token.type != TokenType.KEYWORD or token.value != Keyword.END:
            raise_error(f'Expected `end` at the end of memory definition but found: `{token.value}`',
                        token.loc)
        if len(memory_size_stack) != 1:
            raise_error('Memory definition expects only 1 integer', token.loc)
        assert type(memory_name.value) == str, 'Memory name value must be a string'
        return memory_name.value, memory_size_stack.pop()

    def check_block_name_validity(self, token: Token) -> None | NoReturn:
        if type(token.value) == Keyword:
            raise_error(f'Redefinition of keyword: `{token.value.name.lower()}`', token.loc)
        if token.type != TokenType.WORD or type(token.value) != str:
            raise_error(f'Expected block name to be: `word`, but found: `{token.name}`', token.loc)
        if token.value in INTRINSIC_NAMES:
            raise_error(f'Redefinition of intrinsic word: `{token.value}`', token.loc)
        if token.value in self.macros:
            notify_user(f'Macro `{token.value}` was defined at this location', self.macros[token.value].loc)
            raise_error(f'Redefinition of existing macro: `{token.value}`', token.loc)
        if token.value in self.memories:
            raise_error(f'Redefinition of existing memory: `{token.value}`', token.loc)
        return None

    def parse_token_as_op(self, token: Token) -> Op | NoReturn:
        assert len(OpType) == 13, 'Exhaustive handling of built-in words'
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
            return self.parse_keyword(token)
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
        if word.strip('"')[-2:] == '\\0':
            return Token(type=TokenType.STR,
                         value=word.strip('"').encode('utf-8').decode('unicode_escape') + '\0',
                         loc=location,
                         name='string')
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


def generate_program_control_flow(program: Program, filename: str) -> None:
    dotfile = open(filename.replace('.phtn', '.dot'), 'w')
    write_base = write_indent(dotfile, 0)
    write_level1 = write_indent(dotfile, 1)
    write_base('digraph ControlFlow {')
    write_level1('size="8@8!"')
    write_level1('center=true')
    write_level1('ratio=fill')
    assert len(OpType) == 9, 'Exhaustive handling of op types in control flow'
    i = 0
    for i in range(len(program.ops)):
        op = program.ops[i]
        if op.type == OpType.PUSH_INT:
            write_level1(f'Node_{i} [label={op.operand}]')
            write_level1(f'Node_{i} -> Node_{i + 1}')
        elif op.type == OpType.PUSH_STR:
            write_level1(f'Node_{i} [label={repr(repr(op.operand))}]')
            write_level1(f'Node_{i} -> Node_{i + 1}')
        elif op.type == OpType.INTRINSIC:
            assert type(op.operand) == Intrinsic, 'Operand must be an intrinsic'
            write_level1(f'Node_{i} [label={op.operand.name}]')
            write_level1(f'Node_{i} -> Node_{i + 1}')
        elif op.type == OpType.IF:
            assert type(op.token.value) == Keyword, 'Token value must be a keyword'
            write_level1(f'Node_{i} [label={op.token.value.name} shape=octagon]')
            write_level1(f'Node_{i} -> Node_{i + 1}')
        elif op.type == OpType.ELIF:
            assert type(op.token.value) == Keyword, 'Token value must be a keyword'
            assert type(op.operand) == int, 'Operand must be an integer'
            write_level1(f'Node_{i} [label={op.token.value.name}]')
            write_level1(f'Node_{i} -> Node_{op.operand + 1}')
        elif op.type == OpType.ELSE:
            assert type(op.token.value) == Keyword, 'Token value must be a keyword'
            assert type(op.operand) == int, 'Operand must be an integer'
            write_level1(f'Node_{i} [label={op.token.value.name}]')
            write_level1(f'Node_{i} -> Node_{op.operand + 1}')
        elif op.type == OpType.WHILE:
            assert type(op.token.value) == Keyword, 'Token value must be a keyword'
            write_level1(f'Node_{i} [label={op.token.value.name} shape=octagon]')
            write_level1(f'Node_{i} -> Node_{i + 1}')
        elif op.type == OpType.DO:
            assert type(op.token.value) == Keyword, 'Token value must be a keyword'
            assert type(op.operand) == int, 'Operand must be an integer'
            write_level1(f'Node_{i} [label={op.token.value.name} shape=diamond]')
            write_level1(f'Node_{i} -> Node_{i + 1} [label=True]')
            write_level1(f'Node_{i} -> Node_{op.operand + 1} [label=False style=dashed]')
        elif op.type == OpType.END:
            assert type(op.token.value) == Keyword, 'Token value must be a keyword'
            write_level1(f'Node_{i} [label={op.token.value.name}]')
            if op.operand is None:
                write_level1(f'Node_{i} -> Node_{i + 1}')
            else:
                write_level1(f'Node_{i} -> Node_{op.operand}')
        else:
            assert False, f'Unhandled op type: {op.type}'
    write_level1(f'Node_{i + 1} [label=Halt shape=box]')
    write_base('}')
    dotfile.close()


if __name__ == '__main__':
    _, *argv = sys.argv
    if len(argv) < 2:
        usage_help()
        exit(1)
    subcommand, *argv = argv
    if subcommand not in ('sim', 'com', 'flow'):
        usage_help()
        exit(1)
    debug_arg = '-d' in argv or '--debug' in argv
    unsafe_arg = '--unsafe' in argv
    file_path_arg = argv[0]
    program_stack = lex_file(file_path_arg)
    token_parser = Parsing(program_stack)
    program_referenced = token_parser.parse_tokens_to_program()
    if not unsafe_arg:
        type_check_program(program_referenced, debug_arg)
    if subcommand == 'sim':
        simulate_little_endian_macos(program_referenced, argv)
    elif subcommand == 'flow':
        generate_program_control_flow(program_referenced, file_path_arg)
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

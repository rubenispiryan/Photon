import inspect
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generator, List, NoReturn, Callable, Dict, TextIO

MACRO_EXPANSION_LIMIT = 100_000

@dataclass
class Loc:
    filename: str
    line: int
    col: int


def make_log_message(message: str, loc: Loc) -> str:
    return f'{loc.filename}:{loc.line + 1}:{loc.col + 1}: {message}'


def notify_user(message: str, loc: Loc) -> None:
    print(make_log_message('[NOTE] ' + message, loc))


def raise_error(message: str, loc: Loc) -> NoReturn:
    current_frame = inspect.currentframe()
    caller_frame = inspect.getouterframes(current_frame, 2)
    caller_info = caller_frame[1]
    caller_function = caller_info.function
    trace_message = f'Error message originated inside: {caller_function}'
    notify_user(trace_message, Loc(filename=os.path.abspath(caller_info.filename),
                                   line=caller_info.lineno - 1,
                                   col=0))

    print(make_log_message('[ERROR] ' + message, loc))
    exit(1)


def write_indent(file: TextIO, level: int = 0) -> Callable[[str], None]:
    def temp(buffer: str) -> None:
        file.write(' ' * (0 if level == 0 else 4 ** level) + buffer + '\n')

    return temp


def asm_setup(write_base: Callable[[str], None], write_level1: Callable[[str], None]) -> None:
    write_base('.macro push reg1:req')
    write_level1(r'str \reg1, [sp, #-16]!')
    write_base('.endmacro')
    write_base('.macro pop reg1:req')
    write_level1(r'ldr \reg1, [sp], #16')
    write_base('.endmacro')
    write_base('.macro pushw reg1')
    write_level1('sub sp, sp, #16')
    write_level1(r'stp \reg1, wzr, [sp]')
    write_base('.endmacro')
    write_base('.macro popw reg1')
    write_level1(r'ldp \reg1, wzr, [sp]')
    write_level1('add sp, sp, #16')
    write_base('.endmacro')
    write_base('dump:')
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
    OP_EQUAL = auto()
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    NE = auto()
    MEM = auto()
    LOAD = auto()
    STORE = auto()
    LOAD64 = auto()
    STORE64 = auto()
    SYSCALL1 = auto()
    SYSCALL3 = auto()
    DUP = auto()
    DROP = auto()
    BITAND = auto()
    BITOR = auto()
    SHIFT_RIGHT = auto()
    SHIFT_LEFT = auto()
    SWAP = auto()
    OVER = auto()


class OpType(Enum):
    PUSH_INT = auto()
    PUSH_STR = auto()
    INTRINSIC = auto()
    IF = auto()
    END = auto()
    ELSE = auto()
    WHILE = auto()
    DO = auto()


@dataclass
class Op:
    type: OpType
    loc: Loc
    name: str
    operand: int | str | Intrinsic | None = None
    addr: int | None = None


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


@dataclass
class Token:
    type: TokenType
    value: int | str | Keyword
    loc: Loc
    name: str


@dataclass
class Macro:
    tokens: List[Token]
    loc: Loc
    expand_count: int = 0

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
    '==': Intrinsic.OP_EQUAL,
    '>': Intrinsic.GT,
    '<': Intrinsic.LT,
    'dup': Intrinsic.DUP,
    'mem': Intrinsic.MEM,
    '.': Intrinsic.STORE,
    ',': Intrinsic.LOAD,
    '.64': Intrinsic.STORE64,
    ',64': Intrinsic.LOAD64,
    'syscall1': Intrinsic.SYSCALL1,
    'syscall3': Intrinsic.SYSCALL3,
    'drop': Intrinsic.DROP,
    '&': Intrinsic.BITAND,
    '|': Intrinsic.BITOR,
    '>>': Intrinsic.SHIFT_RIGHT,
    '<<': Intrinsic.SHIFT_LEFT,
    'swap': Intrinsic.SWAP,
    'over': Intrinsic.OVER,
    '>=': Intrinsic.GTE,
    '<=': Intrinsic.LTE,
    '!=': Intrinsic.NE,
}

assert len(INTRINSIC_NAMES) == len(Intrinsic), 'Exhaustive handling of intrinsics'

STR_CAPACITY = 640_000
MEM_CAPACITY = 640_000


def simulate_program(program: List[Op]) -> None:
    stack: List = []
    assert len(OpType) == 8, 'Exhaustive handling of operators in simulation'
    i = 0
    mem = bytearray(STR_CAPACITY + MEM_CAPACITY)
    str_size = 0
    allocated_strs = {}
    while i < len(program):
        operation = program[i]
        try:
            if operation.type == OpType.PUSH_INT:
                assert type(operation.operand) == int, 'Value for `PUSH_INT` must be `int`'
                stack.append(operation.operand)
            elif operation.type == OpType.PUSH_STR:
                assert type(operation.operand) == str, 'Value for `PUSH_STR` must be `str`'
                bs = bytes(operation.operand, 'utf-8')
                n = len(bs)
                stack.append(n)
                if operation.operand not in allocated_strs:
                    allocated_strs[operation.operand] = str_size
                    mem[str_size:str_size + n] = bs
                    str_size += n
                    if str_size > STR_CAPACITY:
                        raise_error('String buffer overflow', operation.loc)
                stack.append(allocated_strs[operation.operand])
            elif operation.type == OpType.IF:
                a = stack.pop()
                if a == 0:
                    assert type(operation.operand) == int, 'Jump address must be `int`'
                    i = operation.operand
            elif operation.type == OpType.ELSE:
                assert type(operation.operand) == int, 'Jump address must be `int`'
                i = operation.operand
            elif operation.type == OpType.END:
                if operation.operand is not None:
                    assert type(operation.operand) == int, 'Jump address must be `int`'
                    i = operation.operand
            elif operation.type == OpType.WHILE:
                i += 1
                continue
            elif operation.type == OpType.DO:
                a = stack.pop()
                assert type(a) == int, 'Arguments for `do` must be `int`'
                if a == 0:
                    assert type(operation.operand) == int, 'Jump address must be `int`'
                    i = operation.operand
            elif operation.type == OpType.INTRINSIC:
                assert len(Intrinsic) == 26, 'Exhaustive handling of intrinsics in simulation'
                if operation.operand == Intrinsic.ADD:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `+` must be `int`'
                    stack.append(a + b)
                elif operation.operand == Intrinsic.SUB:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `-` must be `int`'
                    stack.append(b - a)
                elif operation.operand == Intrinsic.MUL:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `*` must be `int`'
                    stack.append(a * b)
                elif operation.operand == Intrinsic.DIV:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `/` must be `int`'
                    stack.append(b // a)
                elif operation.operand == Intrinsic.PRINT:
                    a = stack.pop()
                    assert type(a) == int, 'Arguments for `print` must be `int`'
                    print(a)
                elif operation.operand == Intrinsic.OP_EQUAL:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `==` must be `int`'
                    stack.append(int(a == b))
                elif operation.operand == Intrinsic.LT:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `<` must be `int`'
                    stack.append(int(b < a))
                elif operation.operand == Intrinsic.GT:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `>` must be `int`'
                    stack.append(int(b > a))
                elif operation.operand == Intrinsic.GTE:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `>=` must be `int`'
                    stack.append(int(b >= a))
                elif operation.operand == Intrinsic.LTE:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `<=` must be `int`'
                    stack.append(int(b <= a))
                elif operation.operand == Intrinsic.NE:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `!=` must be `int`'
                    stack.append(int(a != b))
                elif operation.operand == Intrinsic.DUP:
                    a = stack.pop()
                    assert type(a) == int, 'Arguments for `dup` must be `int`'
                    stack.append(a)
                    stack.append(a)
                elif operation.operand == Intrinsic.DROP:
                    stack.pop()
                elif operation.operand == Intrinsic.MEM:
                    stack.append(STR_CAPACITY)
                elif operation.operand == Intrinsic.LOAD:
                    address = stack.pop()
                    assert type(address) == int, 'Arguments for `,` must be `int`'
                    stack.append(mem[address])
                elif operation.operand == Intrinsic.STORE:
                    value = stack.pop()
                    address = stack.pop()
                    assert type(value) == type(address) == int, 'Arguments for `.` must be `int`'
                    mem[address] = value & 0xFF
                elif operation.operand == Intrinsic.LOAD64:
                    addr = stack.pop()
                    _bytes = bytearray(8)
                    for offset in range(0, 8):
                        _bytes[offset] = mem[addr + offset]
                    stack.append(int.from_bytes(_bytes, byteorder="little"))
                elif operation.operand == Intrinsic.STORE64:
                    store_value64 = stack.pop().to_bytes(length=8, byteorder="little")
                    store_addr64 = stack.pop()
                    for byte in store_value64:
                        mem[store_addr64] = byte
                        store_addr64 += 1
                elif operation.operand == Intrinsic.BITOR:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `|` must be `int`'
                    stack.append(a | b)
                elif operation.operand == Intrinsic.BITAND:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `&` must be `int`'
                    stack.append(a & b)
                elif operation.operand == Intrinsic.SHIFT_RIGHT:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `>>` must be `int`'
                    stack.append(b >> a)
                elif operation.operand == Intrinsic.SHIFT_LEFT:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `<<` must be `int`'
                    stack.append(b << a)
                elif operation.operand == Intrinsic.SWAP:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `swap` must be `int`'
                    stack.append(a)
                    stack.append(b)
                elif operation.operand == Intrinsic.OVER:
                    a = stack.pop()
                    b = stack.pop()
                    assert type(a) == type(b) == int, 'Arguments for `over` must be `int`'
                    stack.append(b)
                    stack.append(a)
                    stack.append(b)
                elif operation.operand == Intrinsic.SYSCALL1:
                    syscall_number = stack.pop()
                    arg1 = stack.pop()
                    if syscall_number == 1:
                        exit(arg1)
                elif operation.operand == Intrinsic.SYSCALL3:
                    syscall_number = stack.pop()
                    arg1 = stack.pop()
                    arg2 = stack.pop()
                    arg3 = stack.pop()
                    if syscall_number == 4:
                        assert type(arg1) == type(arg2) == type(arg3) == int, 'Arguments for `syscall3` must be `int`'
                        if arg1 == 1:
                            print(mem[arg2:arg2 + arg3].decode(), end='')
                        elif arg1 == 2:
                            print(mem[arg2:arg2 + arg3].decode(), end='', file=sys.stderr)
                        else:
                            raise_error(f'Unknown file descriptor: {arg1}', operation.loc)
                    else:
                        raise_error(f'Unknown syscall number: {syscall_number}', operation.loc)
                else:
                    raise_error(f'Unhandled intrinsic: {operation.name}',
                                operation.loc)
            else:
                raise_error(f'Unhandled operation: {operation.name}',
                            operation.loc)
        except AssertionError as e:
            raise e
        except Exception as e:
            raise_error(f'Exception in Simulation: {str(e)}', operation.loc)
        i += 1


def compile_program(program: List[Op]) -> None:
    assert len(OpType) == 8, 'Exhaustive handling of operators in compilation'
    out = open('output.s', 'w')
    write_base = write_indent(out, 0)
    write_level1 = write_indent(out, 1)
    write_base('.section __TEXT, __text')
    write_base('.global _start')
    write_base('.align 3')
    asm_setup(write_base, write_level1)
    write_base('_start:')
    strs: List[str] = []
    allocated_strs: Dict[str, int] = {}
    for i in range(len(program)):
        operation = program[i]
        if operation.type == OpType.PUSH_INT:
            assert type(operation.operand) == int, 'Operation value must be an `int` for PUSH_INT'
            write_level1(f'ldr x0, ={operation.operand}')
            write_level1('push x0')
        elif operation.type == OpType.PUSH_STR:
            assert type(operation.operand) == str, 'Operation value must be a `str` for PUSH_STR'
            write_level1(f'ldr x0, ={len(operation.operand)}')
            write_level1('push x0')
            address = allocated_strs.get(operation.operand, len(strs))
            write_level1(f'adrp x1, str_{address}@PAGE')
            write_level1(f'add x1, x1, str_{address}@PAGEOFF')
            write_level1('push x1')
            if operation.operand not in allocated_strs:
                allocated_strs[operation.operand] = len(strs)
                strs.append(operation.operand)
        elif operation.type in (OpType.IF, OpType.DO):
            write_level1('pop x0')
            write_level1('tst x0, x0')
            write_level1(f'b.eq end_{operation.operand}')
        elif operation.type == OpType.ELSE:
            write_level1(f'b end_{operation.operand}')
            write_base(f'end_{i}:')
        elif operation.type == OpType.END:
            if operation.operand is not None:
                write_level1(f'b while_{operation.operand}')
            write_base(f'end_{i}:')
        elif operation.type == OpType.WHILE:
            write_base(f'while_{i}:')
        elif operation.type == OpType.INTRINSIC:
            assert len(Intrinsic) == 26, 'Exhaustive handling of intrinsics in simulation'
            if operation.operand == Intrinsic.ADD:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('add x0, x0, x1')
                write_level1('push x0')
            elif operation.operand == Intrinsic.SUB:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('sub x0, x1, x0')
                write_level1('push x0')
            elif operation.operand == Intrinsic.MUL:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('mul x0, x0, x1')
                write_level1('push x0')
            elif operation.operand == Intrinsic.DIV:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('udiv x0, x1, x0')
                write_level1('push x0')
            elif operation.operand == Intrinsic.PRINT:
                write_level1('pop x0')
                write_level1('bl dump')
            elif operation.operand == Intrinsic.OP_EQUAL:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('cmp x0, x1')
                write_level1('cset x0, eq')
                write_level1('push x0')
            elif operation.operand == Intrinsic.LT:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('cmp x1, x0')
                write_level1('cset x0, lt')
                write_level1('push x0')
            elif operation.operand == Intrinsic.GT:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('cmp x1, x0')
                write_level1('cset x0, gt')
                write_level1('push x0')
            elif operation.operand == Intrinsic.LTE:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('cmp x1, x0')
                write_level1('cset x0, le')
                write_level1('push x0')
            elif operation.operand == Intrinsic.GTE:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('cmp x1, x0')
                write_level1('cset x0, ge')
                write_level1('push x0')
            elif operation.operand == Intrinsic.NE:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('cmp x0, x1')
                write_level1('cset x0, ne')
                write_level1('push x0')
            elif operation.operand == Intrinsic.DUP:
                write_level1('pop x0')
                write_level1('push x0')
                write_level1('push x0')
            elif operation.operand == Intrinsic.DROP:
                write_level1('pop x0')
            elif operation.operand == Intrinsic.MEM:
                write_level1('adrp x0, mem@PAGE')
                write_level1('add x0, x0, mem@PAGEOFF')
                write_level1('push x0')
            elif operation.operand == Intrinsic.STORE:
                write_level1('popw w0')
                write_level1('pop x1')
                write_level1('strb w0, [x1]')
            elif operation.operand == Intrinsic.LOAD:
                write_level1('pop x0')
                write_level1('ldrb w1, [x0]')
                write_level1('pushw w1')
            elif operation.operand == Intrinsic.STORE64:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('str x0, [x1]')
            elif operation.operand == Intrinsic.LOAD64:
                write_level1('pop x0')
                write_level1('ldr x1, [x0]')
                write_level1('push x1')
            elif operation.operand == Intrinsic.BITOR:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('orr x0, x0, x1')
                write_level1('push x0')
            elif operation.operand == Intrinsic.BITAND:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('and x0, x0, x1')
                write_level1('push x0')
            elif operation.operand == Intrinsic.SHIFT_RIGHT:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('lsr x0, x1, x0')
                write_level1('push x0')
            elif operation.operand == Intrinsic.SHIFT_LEFT:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('lsl x0, x1, x0')
                write_level1('push x0')
            elif operation.operand == Intrinsic.SWAP:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('push x0')
                write_level1('push x1')
            elif operation.operand == Intrinsic.OVER:
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('push x1')
                write_level1('push x0')
                write_level1('push x1')
            elif operation.operand == Intrinsic.SYSCALL1:
                write_level1('pop x16')
                write_level1('pop x0')
                write_level1('svc #0')
            elif operation.operand == Intrinsic.SYSCALL3:
                write_level1('pop x16')
                write_level1('pop x0')
                write_level1('pop x1')
                write_level1('pop x2')
                write_level1('svc #0')
            else:
                raise_error(f'Unhandled intrinsic: {operation.name}',
                            operation.loc)
        else:
            raise_error(f'Unhandled operation: {operation.name}',
                        operation.loc)
    write_level1('mov x16, #1')
    write_level1('mov x0, #0')
    write_level1('svc #0')
    write_base('.section __DATA, __data')
    for i in range(len(strs)):
        word = repr(strs[i]).strip("'")
        write_level1(f'str_{i}: .ascii "{word}"')
    write_base('.section __DATA, __bss')
    write_base('mem:')
    write_level1(f'.skip {MEM_CAPACITY}')
    out.close()


def usage_help() -> None:
    print('Usage: photon.py <SUBCOMMAND> <FLAGS> <FILENAME>')
    print('Subcommands:')
    print('     sim     Simulate the program')
    print('     com     Compile the program')
    print('         --run   Used with `com` to run immediately')


def parse_keyword(stack: List[int], token: Token, i: int, program: List[Op]) -> NoReturn | Op:
    assert len(Keyword) == 7, 'Exhaustive handling of keywords in parse_keyword'
    if type(token.value) != Keyword:
        raise_error(f'Token value `{token.value}` must be a Keyword, but found: {type(token.value)}', token.loc)
    if token.value == Keyword.IF:
        stack.append(i)
        return Op(type=OpType.IF, loc=token.loc, name=token.name)
    elif token.value == Keyword.ELSE:
        if_index = stack.pop()
        if program[if_index].type != OpType.IF:
            raise_error(f'Else can only be used with an `if`', program[if_index].loc)
        program[if_index].operand = i
        stack.append(i)
        return Op(type=OpType.ELSE, loc=token.loc, name=token.name)
    elif token.value == Keyword.WHILE:
        stack.append(i)
        return Op(type=OpType.WHILE, loc=token.loc, name=token.name)
    elif token.value == Keyword.DO:
        stack.append(i)
        return Op(type=OpType.DO, loc=token.loc, name=token.name)
    elif token.value == Keyword.END:
        block_index = stack.pop()
        if program[block_index].type in (OpType.IF, OpType.ELSE):
            program[block_index].operand = i
            return Op(type=OpType.END, loc=token.loc, name=token.name)
        elif program[block_index].type == OpType.DO:
            program[block_index].operand = i
            while_index = stack.pop()
            if program[while_index].type != OpType.WHILE:
                raise_error('`while` must be present before `do`', program[while_index].loc)
            value = while_index
            return Op(type=OpType.END, loc=token.loc, name=token.name, operand=value)
        else:
            raise_error('End can only be used with an `if`, `else`, `while` or `macro`',
                        program[block_index].loc)
    else:
        raise_error(f'Unknown keyword token: {token.value}', token.loc)


def expand_keyword_to_tokens(token: Token, rprogram: List[Token], macros: Dict[str, Macro]) -> NoReturn | None:
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
            raise_error(f'Redefinition of existing macro: `{macro_name.value}`\n', macro_name.loc)
        if len(rprogram) == 0:
            raise_error(f'Expected `end` at the end of empty macro definition but found: `{macro_name.value}`',
                        macro_name.loc)
        macros[macro_name.value] = Macro([], token.loc)
        block_count = 0
        while len(rprogram) > 0:
            next_token = rprogram.pop()
            if next_token.type == TokenType.KEYWORD and next_token.value == Keyword.END:
                if block_count == 0:
                    break
                block_count -= 1
            elif next_token.type == TokenType.KEYWORD and next_token.value in (Keyword.MACRO, Keyword.IF, Keyword.WHILE):
                block_count += 1
            macros[macro_name.value].tokens.append(next_token)
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
                lexed_include = lex_file(os.path.abspath(include_filepath))
                rprogram.extend(reversed(lexed_include))
                found = True
        if not found:
            raise_error(f'Photon file with name: {include_name.value} not found', include_name.loc)
    else:
        raise_error(f'Keyword token not compilable to tokens: {token.value}', token.loc)
    return None


def compile_tokens_to_program(token_program: List[Token]) -> List[Op]:
    assert len(TokenType) == 5, "Exhaustive handling of tokens in compile_tokens_to_program."
    stack: List[int] = []
    rprogram = list(reversed(token_program))
    program: List[Op] = []
    macros: Dict[str, Macro] = {}
    i = 0
    while len(rprogram) > 0:
        token = rprogram.pop()
        if token.value in macros:
            assert type(token.value) == str, 'Compiler Error: non string macro name was saved'
            current_macro = macros[token.value]
            current_macro.expand_count += 1
            if current_macro.expand_count > MACRO_EXPANSION_LIMIT:
                raise_error(f'Expansion limit reached for macro: {token.value}', current_macro.loc)
            rprogram.extend(reversed(current_macro.tokens))
            continue
        if token.type == TokenType.KEYWORD and token.value in (Keyword.MACRO, Keyword.INCLUDE):
            expand_keyword_to_tokens(token, rprogram, macros)
        else:
            program.append(parse_token_as_op(stack, token, i, program))
            i += 1
    return program


def parse_token_as_op(stack: List[int], token: Token, i: int, program: List[Op]) -> Op | NoReturn:
    assert len(OpType) == 8, 'Exhaustive handling of built-in words'
    assert len(TokenType) == 5, 'Exhaustive handling of tokens in parser'

    if token.type == TokenType.INT:
        if type(token.value) != int:
            raise_error('Token value must be an integer', token.loc)
        return Op(type=OpType.PUSH_INT, operand=token.value, loc=token.loc, name=token.name)
    elif token.type == TokenType.CHAR:
        if type(token.value) != int:
            raise_error('Token value must be an integer', token.loc)
        return Op(type=OpType.PUSH_INT, operand=token.value, loc=token.loc, name=token.name)
    elif token.type == TokenType.STR:
        if type(token.value) != str:
            raise_error('Token value must be an string', token.loc)
        return Op(type=OpType.PUSH_STR, operand=token.value, loc=token.loc, name=token.name)
    elif token.type == TokenType.KEYWORD:
        return parse_keyword(stack, token, i, program)
    elif token.type == TokenType.WORD:
        assert type(token.value) == str, "`word` must be a string"
        if token.value not in INTRINSIC_NAMES:
            raise_error(f'Unknown intrinsic name: `{token.value}`', token.loc)
        return Op(type=OpType.INTRINSIC, operand=INTRINSIC_NAMES[token.value], loc=token.loc, name=token.value)
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
    _, *argv = sys.argv
    if len(argv) < 2:
        usage_help()
        exit(1)
    subcommand, *argv = argv
    filename_arg, *argv = argv
    file_path_arg = os.path.abspath(filename_arg)
    program_stack = lex_file(file_path_arg)
    program_referenced = compile_tokens_to_program(program_stack)
    if subcommand == 'sim':
        simulate_program(program_referenced)
    elif subcommand == 'com':
        compile_program(program_referenced)
        exit_code = subprocess.call('as -o output.o output.s', shell=True)
        if exit_code != 0:
            exit(exit_code)
        exit_code = subprocess.call(
            'ld -o output output.o -lSystem -syslibroot `xcrun -sdk macosx'
            ' --show-sdk-path` -e _start -arch arm64', shell=True)
        if exit_code != 0:
            exit(exit_code)
        if '--run' in argv:
            exit(subprocess.call('./output', shell=True))
    else:
        usage_help()
        exit(1)

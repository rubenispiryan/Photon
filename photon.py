import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum, auto
from io import FileIO
from typing import Tuple, Generator, List, NoReturn, Callable, Dict, TextIO


def raise_error(message: str, loc: Tuple[str, int, int]) -> NoReturn:
    print(f'{loc[0]}:{loc[1] + 1}:{loc[2]}: {message}')
    exit(1)


def write_indent(file: TextIO, level:int = 0) -> Callable[[str], None]:
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


class OpType(Enum):
    PUSH_INT = auto()
    PUSH_STR = auto()
    ADD = auto()
    SUB = auto()
    PRINT = auto()
    OP_EQUAL = auto()
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    NE = auto()
    IF = auto()
    END = auto()
    ELSE = auto()
    WHILE = auto()
    DO = auto()
    MEM = auto()
    LOAD = auto()
    STORE = auto()
    SYSCALL3 = auto()
    DUP = auto()
    DUP2 = auto()
    DROP = auto()
    DROP2 = auto()
    BITAND = auto()
    BITOR = auto()
    SHIFT_RIGHT = auto()
    SHIFT_LEFT = auto()
    SWAP = auto()
    OVER = auto()
    MOD = auto()


@dataclass
class Op:
    type: OpType
    loc: Tuple[str, int, int]
    value: int | str | None = None
    jmp: int | None = None
    addr: int | None = None


class TokenType(Enum):
    WORD = auto()
    INT = auto()
    STR = auto()

@dataclass
class Token:
    type: TokenType
    value: int | str
    loc: Tuple[str, int, int]


BUILTIN_NAMES = {
    OpType.PRINT: 'print',
    OpType.ADD: '+',
    OpType.SUB: '-',
    OpType.OP_EQUAL: '==',
    OpType.GT: '>',
    OpType.LT: '<',
    OpType.IF: 'if',
    OpType.END: 'end',
    OpType.ELSE: 'else',
    OpType.DUP: 'dup',
    OpType.WHILE: 'while',
    OpType.DO: 'do',
    OpType.MEM: 'mem',
    OpType.STORE: '.',
    OpType.LOAD: ',',
    OpType.SYSCALL3: 'syscall3',
    OpType.DUP2: 'dup2',
    OpType.DROP: 'drop',
    OpType.BITAND: '&',
    OpType.BITOR: '|',
    OpType.SHIFT_RIGHT: '<<',
    OpType.SHIFT_LEFT: '>>',
    OpType.SWAP: 'swap',
    OpType.OVER: 'over',
    OpType.DROP2: 'drop2',
    OpType.MOD: '%',
    OpType.GTE: '>=',
    OpType.LTE: '<=',
    OpType.NE: '!=',
}

assert len(OpType) == len(BUILTIN_NAMES) + 2, 'Exhaustive handling of built-in word names'

STR_CAPACITY = 640_000
MEM_CAPACITY = 640_000


def simulate_program(program: List[Op]) -> None:
    stack = []
    assert len(OpType) == 31, 'Exhaustive handling of operators in simulation'
    i = 0
    mem = bytearray(STR_CAPACITY + MEM_CAPACITY)
    str_size = 0
    allocated_strs = {}
    while i < len(program):
        instruction = program[i]
        try:
            if instruction.type == OpType.PUSH_INT:
                stack.append(instruction.value)
            elif instruction.type == OpType.PUSH_STR:
                assert type(instruction.value) == str, 'Value for `PUSH_STR` must be `str`'
                bs = bytes(instruction.value, 'utf-8')
                n = len(bs)
                stack.append(n)
                if instruction.value not in allocated_strs:
                    allocated_strs[instruction.value] = str_size
                    mem[str_size:str_size + n] = bs
                    str_size += n
                    if str_size > STR_CAPACITY:
                        raise_error('String buffer overflow', instruction.loc)
                stack.append(allocated_strs[instruction.value])
            elif instruction.type == OpType.ADD:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `+` must be `int`'
                stack.append(a + b)
            elif instruction.type == OpType.SUB:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `-` must be `int`'
                stack.append(b - a)
            elif instruction.type == OpType.PRINT:
                a = stack.pop()
                assert type(a) == int, 'Arguments for `print` must be `int`'
                print(a)
            elif instruction.type == OpType.OP_EQUAL:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `==` must be `int`'
                stack.append(int(a == b))
            elif instruction.type == OpType.LT:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `<` must be `int`'
                stack.append(int(b < a))
            elif instruction.type == OpType.GT:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `>` must be `int`'
                stack.append(int(b > a))
            elif instruction.type == OpType.GTE:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `>=` must be `int`'
                stack.append(int(b >= a))
            elif instruction.type == OpType.LTE:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `<=` must be `int`'
                stack.append(int(b <= a))
            elif instruction.type == OpType.NE:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `!=` must be `int`'
                stack.append(int(a != b))
            elif instruction.type == OpType.IF:
                a = stack.pop()
                if a == 0:
                    assert type(instruction.jmp) == int, 'Jump address must be `int`'
                    i = instruction.jmp
            elif instruction.type == OpType.ELSE:
                assert type(instruction.jmp) == int, 'Jump address must be `int`'
                i = instruction.jmp
            elif instruction.type == OpType.END:
                if instruction.jmp is not None:
                    i = instruction.jmp
            elif instruction.type == OpType.DUP:
                a = stack.pop()
                assert type(a) == int, 'Arguments for `dup` must be `int`'
                stack.append(a)
                stack.append(a)
            elif instruction.type == OpType.DUP2:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `dup2` must be `int`'
                stack.append(b)
                stack.append(a)
                stack.append(b)
                stack.append(a)
            elif instruction.type == OpType.DROP:
                stack.pop()
            elif instruction.type == OpType.DROP2:
                stack.pop()
                stack.pop()
            elif instruction.type == OpType.WHILE:
                i += 1
                continue
            elif instruction.type == OpType.DO:
                a = stack.pop()
                assert type(a) ==  int, 'Arguments for `do` must be `int`'
                if a == 0:
                    assert type(instruction.jmp) == int, 'Jump address must be `int`'
                    i = instruction.jmp
            elif instruction.type == OpType.MEM:
                stack.append(0)
            elif instruction.type == OpType.LOAD:
                address = stack.pop()
                assert type(address) == int, 'Arguments for `,` must be `int`'
                stack.append(mem[address])
            elif instruction.type == OpType.STORE:
                value = stack.pop()
                address = stack.pop()
                assert type(value) == type(address) == int, 'Arguments for `.` must be `int`'
                mem[address] = value & 0xFF
            elif instruction.type == OpType.BITOR:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `|` must be `int`'
                stack.append(a | b)
            elif instruction.type == OpType.BITAND:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `&` must be `int`'
                stack.append(a & b)
            elif instruction.type == OpType.SHIFT_RIGHT:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `>>` must be `int`'
                stack.append(b >> a)
            elif instruction.type == OpType.SHIFT_LEFT:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `<<` must be `int`'
                stack.append(b << a)
            elif instruction.type == OpType.SWAP:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `swap` must be `int`'
                stack.append(a)
                stack.append(b)
            elif instruction.type == OpType.OVER:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `over` must be `int`'
                stack.append(b)
                stack.append(a)
                stack.append(b)
            elif instruction.type == OpType.MOD:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `mod` must be `int`'
                stack.append(b % a)
            elif instruction.type == OpType.SYSCALL3:
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
                        raise_error(f'Unknown file descriptor: {arg1}', instruction.loc)
                else:
                    raise_error(f'Unknown syscall number: {syscall_number}', instruction.loc)
            else:
                raise_error(f'Unhandled instruction: {BUILTIN_NAMES[instruction.type]}',
                            instruction.loc)
        except Exception as e:
            raise_error(f'Exception in Simulation: {str(e)}', instruction.loc)
        i += 1


def compile_program(program: List[Op]) -> None:
    assert len(OpType) == 31, 'Exhaustive handling of operators in compilation'
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
        instruction = program[i]
        if instruction.type == OpType.PUSH_INT:
            assert type(instruction.value) == int, 'Instruction value must be an `int` for PUSH_INT'
            write_level1(f'ldr x0, ={instruction.value}')
            write_level1('push x0')
        elif instruction.type == OpType.PUSH_STR:
            assert type(instruction.value) == str, 'Instruction value must be a `str` for PUSH_STR'
            write_level1(f'ldr x0, ={len(instruction.value)}')
            write_level1('push x0')
            address = allocated_strs.get(instruction.value, len(strs))
            write_level1(f'adrp x1, str_{address}@PAGE')
            write_level1(f'add x1, x1, str_{address}@PAGEOFF')
            write_level1('push x1')
            if instruction.value not in allocated_strs:
                allocated_strs[instruction.value] = len(strs)
                strs.append(instruction.value)
        elif instruction.type == OpType.ADD:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('add x0, x0, x1')
            write_level1('push x0')
        elif instruction.type == OpType.SUB:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('sub x0, x1, x0')
            write_level1('push x0')
        elif instruction.type == OpType.PRINT:
            write_level1('pop x0')
            write_level1('bl dump')
        elif instruction.type == OpType.OP_EQUAL:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('cmp x0, x1')
            write_level1('cset x0, eq')
            write_level1('push x0')
        elif instruction.type == OpType.LT:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('cmp x1, x0')
            write_level1('cset x0, lt')
            write_level1('push x0')
        elif instruction.type == OpType.GT:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('cmp x1, x0')
            write_level1('cset x0, gt')
            write_level1('push x0')
        elif instruction.type == OpType.LTE:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('cmp x1, x0')
            write_level1('cset x0, le')
            write_level1('push x0')
        elif instruction.type == OpType.GTE:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('cmp x1, x0')
            write_level1('cset x0, ge')
            write_level1('push x0')
        elif instruction.type == OpType.NE:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('cmp x0, x1')
            write_level1('cset x0, ne')
            write_level1('push x0')
        elif instruction.type in (OpType.IF, OpType.DO):
            write_level1('pop x0')
            write_level1('tst x0, x0')
            write_level1(f'b.eq end_{instruction.jmp}')
        elif instruction.type == OpType.ELSE:
            write_level1(f'b end_{instruction.jmp}')
            write_base(f'end_{i}:')
        elif instruction.type == OpType.END:
            if instruction.jmp is not None:
                write_level1(f'b while_{instruction.jmp}')
            write_base(f'end_{i}:')
        elif instruction.type == OpType.DUP:
            write_level1('pop x0')
            write_level1('push x0')
            write_level1('push x0')
        elif instruction.type == OpType.DUP2:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('push x1')
            write_level1('push x0')
            write_level1('push x1')
            write_level1('push x0')
        elif instruction.type == OpType.DROP:
            write_level1('pop x0')
        elif instruction.type == OpType.DROP2:
            write_level1('pop x0')
            write_level1('pop x0')
        elif instruction.type == OpType.WHILE:
            write_base(f'while_{i}:')
        elif instruction.type == OpType.MEM:
            write_level1('adrp x0, mem@PAGE')
            write_level1('add x0, x0, mem@PAGEOFF')
            write_level1('push x0')
        elif instruction.type == OpType.STORE:
            write_level1('popw w0')
            write_level1('pop x1')
            write_level1('strb w0, [x1]')
        elif instruction.type == OpType.LOAD:
            write_level1('pop x0')
            write_level1('ldrb w1, [x0]')
            write_level1('pushw w1')
        elif instruction.type == OpType.BITOR:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('orr x0, x0, x1')
            write_level1('push x0')
        elif instruction.type == OpType.BITAND:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('and x0, x0, x1')
            write_level1('push x0')
        elif instruction.type == OpType.SHIFT_RIGHT:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('lsr x0, x1, x0')
            write_level1('push x0')
        elif instruction.type == OpType.SHIFT_LEFT:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('lsl x0, x1, x0')
            write_level1('push x0')
        elif instruction.type == OpType.SWAP:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('push x0')
            write_level1('push x1')
        elif instruction.type == OpType.OVER:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('push x1')
            write_level1('push x0')
            write_level1('push x1')
        elif instruction.type == OpType.MOD:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('udiv x2, x1, x0')
            write_level1('msub x0, x2, x0, x1')
            write_level1('push x0')
        elif instruction.type == OpType.SYSCALL3:
            write_level1('pop x16')
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('pop x2')
            write_level1('svc #0')
        else:
            raise_error(f'Unhandled instruction: {BUILTIN_NAMES[instruction.type]}',
                        instruction.loc)
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
    print('Usage: photon.py <SUBCOMMAND> <FILENAME> <FLAGS>')
    print('Subcommands:')
    print('     sim     Simulate the program')
    print('     com     Compile the program')
    print('         --run   Used with `com` to run immediately')


def cross_reference_blocks(program: List[Op]) -> List[Op]:
    assert len(OpType) == 31, 'Exhaustive handling of code block'
    stack = []
    for i in range(len(program)):
        if program[i].type == OpType.IF:
            stack.append(i)
        elif program[i].type == OpType.ELSE:
            if_index = stack.pop()
            if program[if_index].type != OpType.IF:
                raise_error(f'Else can only be used with an `if`', program[if_index].loc)
            program[if_index].jmp = i
            stack.append(i)
        elif program[i].type == OpType.WHILE:
            stack.append(i)
        elif program[i].type == OpType.DO:
            stack.append(i)
        elif program[i].type == OpType.END:
            block_index = stack.pop()
            if program[block_index].type in (OpType.IF, OpType.ELSE):
                program[block_index].jmp = i
            elif program[block_index].type == OpType.DO:
                program[block_index].jmp = i
                while_index = stack.pop()
                if program[while_index].type != OpType.WHILE:
                    raise_error('`while` must be present before `do`', program[while_index].loc)
                program[i].jmp = while_index
            else:
                raise_error('End can only be used with an `if`, `else` or `while`',
                            program[block_index].loc)
    return program


def parse_token(token: Token) -> Op | NoReturn:
    assert len(OpType) == 31, 'Exhaustive handling of built-in words'
    builtin_words = {
        'print': Op(type=OpType.PRINT, loc=token.loc),
        '+': Op(type=OpType.ADD, loc=token.loc),
        '-': Op(type=OpType.SUB, loc=token.loc),
        '==': Op(type=OpType.OP_EQUAL, loc=token.loc),
        '>': Op(type=OpType.GT, loc=token.loc),
        '<': Op(type=OpType.LT, loc=token.loc),
        'if': Op(type=OpType.IF, loc=token.loc),
        'end': Op(type=OpType.END, loc=token.loc),
        'else': Op(type=OpType.ELSE, loc=token.loc),
        'dup': Op(type=OpType.DUP, loc=token.loc),
        'while': Op(type=OpType.WHILE, loc=token.loc),
        'do': Op(type=OpType.DO, loc=token.loc),
        'mem': Op(type=OpType.MEM, loc=token.loc),
        '.': Op(type=OpType.STORE, loc=token.loc),
        ',': Op(type=OpType.LOAD, loc=token.loc),
        'syscall3': Op(type=OpType.SYSCALL3, loc=token.loc),
        'dup2': Op(type=OpType.DUP2, loc=token.loc),
        'drop': Op(type=OpType.DROP, loc=token.loc),
        '&': Op(type=OpType.BITAND, loc=token.loc),
        '|': Op(type=OpType.BITOR, loc=token.loc),
        '>>': Op(type=OpType.SHIFT_RIGHT, loc=token.loc),
        '<<': Op(type=OpType.SHIFT_LEFT, loc=token.loc),
        'swap': Op(type=OpType.SWAP, loc=token.loc),
        'over': Op(type=OpType.OVER, loc=token.loc),
        'drop2': Op(type=OpType.DROP2, loc=token.loc),
        '%': Op(type=OpType.MOD, loc=token.loc),
        '>=': Op(type=OpType.GTE, loc=token.loc),
        '<=': Op(type=OpType.LTE, loc=token.loc),
        '!=': Op(type=OpType.NE, loc=token.loc),
    }
    assert len(TokenType) == 3, "Exhaustive handling of tokens"

    if token.type == TokenType.INT:
        return Op(type=OpType.PUSH_INT, loc=token.loc, value=token.value)
    elif token.type == TokenType.STR:
        return Op(type=OpType.PUSH_STR, loc=token.loc, value=token.value)
    elif token.type == TokenType.WORD:
        assert type(token.value) == str, "`word` must be a string"
        return builtin_words[token.value]
    else:
        raise_error(f'Unhandled token: {token}', token.loc)


def seek_until(line: str, start: int, predicate: Callable[[str], bool]) -> int:
    while start < len(line) and not predicate(line[start]):
        start += 1
    return start


def lex_line(line: str, file_path: str, line_number: int) -> Generator[Op, None, None]:
    col = seek_until(line, 0, lambda x: not x.isspace())
    while col < len(line):
        location = (file_path, line_number, col)
        if line[col] == '"':
            col_end = seek_until(line, col + 1, lambda x: x == '"')
            word = line[col + 1:col_end]
            yield parse_token(Token(type=TokenType.STR, value=word.encode('utf-8').decode('unicode_escape'),
                              loc=location))
            col = seek_until(line, col_end + 1, lambda x: not x.isspace())
        else:
            col_end = seek_until(line, col, lambda x: x.isspace())
            word = line[col:col_end]
            try:
                yield parse_token(Token(type=TokenType.INT, value=int(word),
                                  loc=location))
            except ValueError:
                yield parse_token(Token(type=TokenType.WORD, value=word,
                                  loc=location))
            col = seek_until(line, col_end, lambda x: not x.isspace())


def lex_file(file_path: str) -> List[Op]:
    program: List[Op] = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            program.extend(lex_line(line, file_path, i))
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
    program_referenced = cross_reference_blocks(program_stack)
    if subcommand == 'sim':
        simulate_program(program_stack)
    elif subcommand == 'com':
        compile_program(program_stack)
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

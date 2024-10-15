import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generator, List, NoReturn, Callable, Dict, TextIO


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


class OpType(Enum):
    PUSH_INT = auto()
    PUSH_STR = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
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
    MACRO = auto()
    END_MACRO = auto()
    INCLUDE = auto()
    MEM = auto()
    LOAD = auto()
    STORE = auto()
    SYSCALL3 = auto()
    DUP = auto()
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
    loc: Loc
    name: str
    value: int | str | None = None
    jmp: int | None = None
    addr: int | None = None


class TokenType(Enum):
    WORD = auto()
    INT = auto()
    STR = auto()
    CHAR = auto()


@dataclass
class Token:
    type: TokenType
    value: int | str
    loc: Loc
    name: str

@dataclass
class Macro:
    tokens: List[Token]
    loc: Loc


BUILTIN_WORDS = {
    'print': OpType.PRINT,
    '+': OpType.ADD,
    '-': OpType.SUB,
    '*': OpType.MUL,
    '==': OpType.OP_EQUAL,
    '>': OpType.GT,
    '<': OpType.LT,
    'if': OpType.IF,
    'end': OpType.END,
    'else': OpType.ELSE,
    'dup': OpType.DUP,
    'while': OpType.WHILE,
    'do': OpType.DO,
    'mem': OpType.MEM,
    '.': OpType.STORE,
    ',': OpType.LOAD,
    'syscall3': OpType.SYSCALL3,
    'drop': OpType.DROP,
    '&': OpType.BITAND,
    '|': OpType.BITOR,
    '>>': OpType.SHIFT_RIGHT,
    '<<': OpType.SHIFT_LEFT,
    'swap': OpType.SWAP,
    'over': OpType.OVER,
    'drop2': OpType.DROP2,
    '%': OpType.MOD,
    '>=': OpType.GTE,
    '<=': OpType.LTE,
    '!=': OpType.NE,
    'macro': OpType.MACRO,
    'endmacro': OpType.END_MACRO,
    'include': OpType.INCLUDE,
}

STR_CAPACITY = 640_000
MEM_CAPACITY = 640_000


def simulate_program(program: List[Op]) -> None:
    stack = []
    assert len(OpType) == 34, 'Exhaustive handling of operators in simulation'
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
            elif instruction.type == OpType.MUL:
                a = stack.pop()
                b = stack.pop()
                assert type(a) == type(b) == int, 'Arguments for `*` must be `int`'
                stack.append(a * b)
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
                assert type(a) == int, 'Arguments for `do` must be `int`'
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
                raise_error(f'Unhandled instruction: {instruction.name}',
                            instruction.loc)
        except Exception as e:
            raise_error(f'Exception in Simulation: {str(e)}', instruction.loc)
        i += 1


def compile_program(program: List[Op]) -> None:
    assert len(OpType) == 34, 'Exhaustive handling of operators in compilation'
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
        elif instruction.type == OpType.MUL:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('mul x0, x0, x1')
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
        elif instruction.type == OpType.MACRO:
            continue
        else:
            raise_error(f'Unhandled instruction: {instruction.name}',
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

# TODO: cross_reference_blocks is too bloated
def cross_reference_blocks(token_program: List[Token]) -> List[Op]:
    assert len(OpType) == 34, 'Exhaustive handling of code block'
    stack = []
    rprogram = list(reversed(token_program))
    program: List[Op] = []
    macros: Dict[str, Macro] = {}
    i = 0
    while len(rprogram) > 0:
        token = rprogram.pop()
        if token.value in macros:
            assert type(token.value) == str, 'Compiler Error: non string macro name was saved'
            rprogram.extend(reversed(macros[token.value].tokens))
            continue
        else:
            op = parse_token(token)
        program.append(op)
        if op.type == OpType.IF:
            stack.append(i)
        elif op.type == OpType.ELSE:
            if_index = stack.pop()
            if program[if_index].type != OpType.IF:
                for macro in macros.values():
                    print([token.value for token in macro.tokens])
                    print('=' * 100, end='\n' * 3)
                raise_error(f'Else can only be used with an `if`', program[if_index].loc)
            program[if_index].jmp = i
            stack.append(i)
        elif op.type == OpType.WHILE:
            stack.append(i)
        elif op.type == OpType.DO:
            stack.append(i)
        elif op.type == OpType.END:
            block_index = stack.pop()
            if program[block_index].type in (OpType.IF, OpType.ELSE):
                program[block_index].jmp = i
            elif program[block_index].type == OpType.DO:
                program[block_index].jmp = i
                while_index = stack.pop()
                if program[while_index].type != OpType.WHILE:
                    raise_error('`while` must be present before `do`', program[while_index].loc)
                op.jmp = while_index
            else:
                raise_error('End can only be used with an `if`, `else` or `while`',
                            program[block_index].loc)
        elif op.type == OpType.INCLUDE:
            program.pop()
            i -= 1
            if len(rprogram) == 0:
                raise_error('Expected name of the include file but found nothing', op.loc)
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
        elif op.type == OpType.MACRO:
            program.pop()
            i -= 1
            if len(rprogram) == 0:
                raise_error('Expected name of the macro but found nothing', op.loc)
            macro_name = rprogram.pop()
            if macro_name.type != TokenType.WORD or type(macro_name.value) != str:
                raise_error(f'Expected macro name to be: `word`, but found: `{macro_name.name}`', macro_name.loc)
            if macro_name.value in BUILTIN_WORDS:
                raise_error(f'Redefinition of builtin word: `{macro_name.value}`', macro_name.loc)
            if macro_name.value in macros:
                notify_user(f'Macro `{macro_name.value}` was defined at this location', macros[macro_name.value].loc)
                raise_error(f'Redefinition of existing macro: `{macro_name.value}`\n', macro_name.loc)
            if len(rprogram) == 0:
                raise_error(f'Expected `endmacro` at the end of empty macro definition but found: `{macro_name.value}`',
                            macro_name.loc)
            macros[macro_name.value] = Macro([], op.loc)
            rec_macro_count = 0
            while len(rprogram) > 0:
                token = rprogram.pop()
                if token.type == TokenType.WORD and token.value == 'endmacro':
                    if rec_macro_count == 0:
                        break
                    rec_macro_count -= 1
                elif token.type == TokenType.WORD and token.value == 'macro':
                    rec_macro_count += 1
                macros[macro_name.value].tokens.append(token)
            if token.type != TokenType.WORD or token.value != 'endmacro':
                raise_error(f'Expected `endmacro` at the end of macro definition but found: `{token.value}`',
                            token.loc)
        elif op.type == OpType.END_MACRO:
            raise_error('Corresponding macro definition not found for `endmacro`', op.loc)
        i += 1
    return program


def parse_token(token: Token) -> Op | NoReturn:
    assert len(OpType) == 34, 'Exhaustive handling of built-in words'
    assert len(TokenType) == 4, "Exhaustive handling of tokens in parser"

    if token.type == TokenType.INT:
        return Op(type=OpType.PUSH_INT, loc=token.loc, value=token.value, name=token.name)
    elif token.type == TokenType.CHAR:
        return Op(type=OpType.PUSH_INT, loc=token.loc, value=token.value, name=token.name)
    elif token.type == TokenType.STR:
        return Op(type=OpType.PUSH_STR, loc=token.loc, value=token.value, name=token.name)
    elif token.type == TokenType.WORD:
        assert type(token.value) == str, "`word` must be a string"
        if token.value not in BUILTIN_WORDS:
            raise_error(f'Unknown word token `{token.value}`', token.loc)
        return Op(type=BUILTIN_WORDS[token.value], loc=token.loc, name=token.value)
    else:
        raise_error(f'Unhandled token: {token}', token.loc)


def seek_until(line: str, start: int, predicate: Callable[[str], bool]) -> int:
    while start < len(line) and not predicate(line[start]):
        start += 1
    return start


def lex_word(word: str, location: Loc) -> Token:
    assert len(TokenType) == 4, "Exhaustive handling of tokens in lexer"
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
    program_referenced = cross_reference_blocks(program_stack)
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

import os
import subprocess
import sys

auto_counter = 0


def write_indent(file, level=0):
    def temp(buffer):
        file.write(' ' * (0 if level == 0 else 4 ** level) + buffer + '\n')

    return temp


def auto(reset=False):
    global auto_counter
    if reset:
        auto_counter = 0
        return auto_counter
    auto_counter += 1
    return auto_counter


def asm_setup(write_base, write_level1):
    write_base('.macro push reg1, reg2')
    write_level1('sub sp, sp, #16')
    write_level1(r'stp \reg1, \reg2, [sp]')
    write_base('.endmacro')
    write_base('.macro pop reg1, reg2')
    write_level1(r'ldp \reg1, \reg2, [sp]')
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


OP_PUSH = auto(True)
OP_ADD = auto()
OP_SUB = auto()
OP_WRITE = auto()
OP_EQUAL = auto()
OP_LT = auto()
OP_GT = auto()
OP_IF = auto()
OP_END = auto()
OP_ELSE = auto()
OP_DUP = auto()
OP_WHILE = auto()
OP_DO = auto()
OP_COUNTER = auto()


def push(a):
    return OP_PUSH, a


def add():
    return (OP_ADD,)


def sub():
    return (OP_SUB,)


def write():
    return (OP_WRITE,)


def equal():
    return (OP_EQUAL,)


def iff():
    return (OP_IF,)


def end():
    return (OP_END,)

def els():
    return (OP_ELSE,)

def lt():
    return (OP_LT,)

def gt():
    return (OP_GT,)

def dup():
    return (OP_DUP,)

def whil():
    return (OP_WHILE,)

def do():
    return (OP_DO,)


def simulate_program(program):
    stack = []
    assert OP_COUNTER == 13, 'Exhaustive handling of operators in simulation'
    i = 0
    while i < len(program):
        instruction = program[i]
        operator = instruction[0]
        if operator == OP_PUSH:
            stack.append(instruction[1])
        elif operator == OP_ADD:
            a = stack.pop()
            b = stack.pop()
            stack.append(a + b)
        elif operator == OP_SUB:
            a = stack.pop()
            b = stack.pop()
            stack.append(b - a)
        elif operator == OP_WRITE:
            a = stack.pop()
            print(a)
        elif operator == OP_EQUAL:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(a == b))
        elif operator == OP_LT:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(b < a))
        elif operator == OP_GT:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(b > a))
        elif operator == OP_IF:
            a = stack.pop()
            if a == 0:
                i = instruction[1]
        elif operator == OP_ELSE:
            i = instruction[1]
        elif operator == OP_END:
            if len(instruction) > 1:
                i = instruction[1]
        elif operator == OP_DUP:
            a = stack.pop()
            stack.append(a)
            stack.append(a)
        elif operator == OP_WHILE:
            i += 1
            continue
        elif operator == OP_DO:
            a = stack.pop()
            if a == 0:
                i = instruction[1]
        else:
            assert False, f'Unhandled instruction: {instruction}'
        i += 1


def compile_program(program):
    assert OP_COUNTER == 13, 'Exhaustive handling of operators in compilation'
    out = open('output.s', 'w')
    write_base = write_indent(out, 0)
    write_level1 = write_indent(out, 1)
    write_base('.section __TEXT, __text')
    write_base('.global _start')
    write_base('.align 2')
    asm_setup(write_base, write_level1)
    write_base('_start:')
    for i in range(len(program)):
        instruction = program[i]
        operator = instruction[0]
        if operator == OP_PUSH:
            write_level1(f'mov x0, #{instruction[1]}')
            write_level1('push x0, xzr')
        elif operator == OP_ADD:
            write_level1('pop x0, xzr')
            write_level1('pop x1, xzr')
            write_level1('add x0, x0, x1')
            write_level1('push x0, xzr')
        elif operator == OP_SUB:
            write_level1('pop x0, xzr')
            write_level1('pop x1, xzr')
            write_level1('sub x0, x1, x0')
            write_level1('push x0, xzr')
        elif operator == OP_WRITE:
            write_level1('pop x0, xzr')
            write_level1('bl dump')
        elif operator == OP_EQUAL:
            write_level1('pop x0, xzr')
            write_level1('pop x1, xzr')
            write_level1('cmp x0, x1')
            write_level1('cset x0, eq')
            write_level1('push x0, xzr')
        elif operator == OP_LT:
            write_level1('pop x0, xzr')
            write_level1('pop x1, xzr')
            write_level1('cmp x0, x1')
            write_level1('cset x0, gt')
            write_level1('push x0, xzr')
        elif operator == OP_GT:
            write_level1('pop x0, xzr')
            write_level1('pop x1, xzr')
            write_level1('cmp x0, x1')
            write_level1('cset x0, lt')
            write_level1('push x0, xzr')
        elif operator in (OP_IF, OP_DO):
            write_level1('pop x0, xzr')
            write_level1('tst x0, x0')
            write_level1(f'b.eq end_{instruction[1]}')
        elif operator == OP_ELSE:
            write_level1(f'b end_{instruction[1]}')
            write_base(f'end_{i}:')
        elif operator == OP_END:
            if len(instruction) > 1:
                write_level1(f'b while_{instruction[1]}')
            write_base(f'end_{i}:')
        elif operator == OP_DUP:
            write_level1('pop x0, xzr')
            write_level1('push x0, xzr')
            write_level1('push x0, xzr')
        elif operator == OP_WHILE:
            write_base(f'while_{i}:')
        else:
            assert False, f'Unhandled instruction: {instruction}'
    write_level1('mov x16, #1')
    write_level1('mov x0, #0')
    write_level1('svc #0')
    out.close()


def usage_help():
    print('Usage: photon.py <SUBCOMMAND> <FILENAME> <FLAGS>')
    print('Subcommands:')
    print('     sim     Simulate the program')
    print('     com     Compile the program')
    print('         --run   Used with `com` to run immediately')


def parse_token(token, location):
    assert OP_COUNTER == 13, 'Exhaustive handling of tokens'
    filename, line, column = location
    if token == '.':
        return write()
    elif token == '+':
        return add()
    elif token == '-':
        return sub()
    elif token.isdigit() or (token[0] == '-' and token[1:].isdigit()):
        return push(int(token))
    elif token == '==':
        return equal()
    elif token == '>':
        return gt()
    elif token == '<':
        return lt()
    elif token == 'if':
        return iff()
    elif token == 'end':
        return end()
    elif token == 'else':
        return els()
    elif token == 'dup':
        return dup()
    elif token == 'while':
        return whil()
    elif token == 'do':
        return do()
    else:
        print(f'{os.path.abspath(filename)}:{line + 1}:{column}: Unhandled token `{token}`')
        exit(1)


def cross_reference_blocks(program):
    assert OP_COUNTER == 13, 'Exhaustive handling of code block'
    stack = []
    for i in range(len(program)):
        operator = program[i][0]
        if operator == OP_IF:
            stack.append(i)
        elif operator == OP_ELSE:
            if_index = stack.pop()
            assert program[if_index][0] == OP_IF, 'Else can only be used with an `if`'
            program[if_index] = (OP_IF, i)
            stack.append(i)
        elif operator == OP_WHILE:
            stack.append(i)
        elif operator == OP_DO:
            stack.append(i)
        elif operator == OP_END:
            block_index = stack.pop()
            if program[block_index][0] in (OP_IF, OP_ELSE):
                program[block_index] = (program[block_index][0], i)
            elif program[block_index][0] == OP_DO:
                program[block_index] = (program[block_index][0], i)
                while_index = stack.pop()
                assert program[while_index][0] == OP_WHILE, '`while` must be present before `do`'
                program[i] = (OP_END, while_index)
            else:
                assert False, 'Else can only be used with an `if` or `else`'

    return program


def lex_line(line, filename, line_number):
    l, r = 0, 0
    while l < len(line):
        if line[l].isspace():
            l += 1
            continue
        r = l
        while r < len(line) and not line[r].isspace():
            r += 1
        yield parse_token(line[l:r], (filename, line_number, l))
        l = r


def lex_file(filename):
    program = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            program.extend(lex_line(line, filename, i))
    return program


if __name__ == '__main__':
    _, *argv = sys.argv
    if len(argv) < 2:
        usage_help()
        exit(1)
    subcommand, *argv = argv
    filename_arg, *argv = argv

    program_stack = lex_file(filename_arg)
    program_referenced = cross_reference_blocks(program_stack)
    if subcommand == 'sim':
        simulate_program(program_stack)
    elif subcommand == 'com':
        compile_program(program_stack)
        subprocess.call('as -o output.o output.s', shell=True)
        subprocess.call(
            'ld -o output output.o -lSystem -syslibroot `xcrun -sdk macosx'
            ' --show-sdk-path` -e _start -arch arm64', shell=True)
        if '--run' in argv:
            subprocess.call('./output', shell=True)
    else:
        usage_help()
        exit(1)

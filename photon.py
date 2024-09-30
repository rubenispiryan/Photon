import os
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
    write_level1('sub     sp, sp,  #64')
    write_level1('stp     x29, x30, [sp,  #48]')
    write_level1('add     x29, sp,  #48')
    write_level1('stur    x0, [x29,  #-8]')
    write_level1('mov     x8,  #1')
    write_level1('str     x8, [sp]')
    write_level1('ldr     x9, [sp]')
    write_level1('mov     x8,  #32')
    write_level1('subs    x9, x8, x9')
    write_level1('add     x8, sp,  #8')
    write_level1('add     x9, x8, x9')
    write_level1('mov     w8,  #10')
    write_level1('strb    w8, [x9]')
    write_level1('b       .LBB0_1')
    write_base('.LBB0_1:')
    write_level1('ldur    x8, [x29,  #-8]')
    write_level1('mov     x9,  #10')
    write_level1('udiv    x10, x8, x9')
    write_level1('mul     x10, x10, x9')
    write_level1('subs    x8, x8, x10')
    write_level1('add     x8, x8,  #48')
    write_level1('ldr     x11, [sp]')
    write_level1('mov     x10,  #32')
    write_level1('subs    x10, x10, x11')
    write_level1('subs    x11, x10,  #1')
    write_level1('add     x10, sp,  #8')
    write_level1('strb    w8, [x10, x11]')
    write_level1('ldr     x8, [sp]')
    write_level1('add     x8, x8,  #1')
    write_level1('str     x8, [sp]')
    write_level1('ldur    x8, [x29,  #-8]')
    write_level1('udiv    x8, x8, x9')
    write_level1('stur    x8, [x29,  #-8]')
    write_level1('b       .LBB0_2')
    write_base('.LBB0_2:')
    write_level1('ldur    x8, [x29,  #-8]')
    write_level1('subs    x8, x8,  #0')
    write_level1('cset    w8, ne')
    write_level1('tbnz    w8,  #0, .LBB0_1')
    write_level1('b       .LBB0_3')
    write_base('.LBB0_3:')
    write_level1('ldr     x9, [sp]')
    write_level1('mov     x8,  #32')
    write_level1('subs    x9, x8, x9')
    write_level1('add     x8, sp,  #8')
    write_level1('add     x1, x8, x9')
    write_level1('ldr     x2, [sp]')
    write_level1('mov     x0,  #1')
    write_level1('mov     x16,  #4')
    write_level1('svc     #0')
    write_level1('ldp     x29, x30, [sp,  #48]')
    write_level1('add     sp, sp,  #64')
    write_level1('ret')


OP_PUSH = auto(True)
OP_ADD = auto()
OP_WRITE = auto()
OP_COUNTER = auto()

def push(a):
    return OP_PUSH, a

def add():
    return (OP_ADD,)

def write():
    return (OP_WRITE,)


def simulate_program(program):
    stack = []
    assert OP_COUNTER == 3, 'Exhaustive handling of operands in simulation'
    for instruction in program:
        operand = instruction[0]
        if operand == OP_PUSH:
            stack.append(instruction[1])
        elif operand == OP_ADD:
            a = stack.pop()
            b = stack.pop()
            stack.append(a + b)
        elif operand == OP_WRITE:
            a = stack.pop()
            print(a)
        else:
            assert False, 'Unhandled instruction'


def compile_program(program):
    assert OP_COUNTER == 3, 'Exhaustive handling of operands in simulation'
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
        operand = instruction[0]
        if operand == OP_PUSH:
            write_level1('mov x0, #{}'.format(instruction[1]))
            write_level1('push x0, xzr')
        elif operand == OP_ADD:
            write_level1('pop x0, xzr')
            write_level1('pop x1, xzr')
            write_level1('add x0, x0, x1')
            write_level1('push x0, xzr')
        elif operand == OP_WRITE:
            write_level1('pop x0, xzr')
            write_level1('bl dump')
        else:
            assert False, 'Unhandled instruction'
    write_level1('mov x16, #1')
    write_level1('mov x0, #0')
    write_level1('svc #0')
    out.close()


def usage_help():
    print('Usage: photon.py <SUBCOMMAND> <FILENAME>')
    print('Subcommands:')
    print('     sim     Simulate the program')
    print('     com     Compile the program')

def parse_token(token):
    assert OP_COUNTER == 3, 'Exhaustive handling of tokens'
    if token == '.':
        return write()
    elif token == '+':
        return add()
    elif token.isdigit() or (token[0] == '-' and token[1:].isdigit()):
        return push(token)
    else:
        assert False, 'Unhandled token'

def read_program(filename):
    program = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.split()
            for token in line:
                program.append(parse_token(token))
    return program


if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage_help()
        exit(1)
    subcommand = sys.argv[1]

    program_stack = read_program(sys.argv[2])

    if subcommand == 'sim':
        simulate_program(program_stack)
    elif subcommand == 'com':
        compile_program(program_stack)
        os.system('as -o output.o output.s')
        os.system('ld -o output output.o -lSystem -syslibroot `xcrun -sdk macosx --show-sdk-path` -e _start -arch arm64')
        os.system('./output')
    else:
        usage_help()
        exit(1)

import os
import subprocess
import sys

auto_counter = 0


def raise_error(message, loc):
    print(f'{loc[0]}:{loc[1] + 1}:{loc[2]}: {message}')
    exit(1)


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
    write_base('.macro push reg1')
    write_level1('sub sp, sp, #16')
    write_level1(r'stp \reg1, xzr, [sp]')
    write_base('.endmacro')
    write_base('.macro pop reg1')
    write_level1(r'ldp \reg1, xzr, [sp]')
    write_level1('add sp, sp, #16')
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


OP_PUSH = auto(True)
OP_ADD = auto()
OP_SUB = auto()
OP_PRINT = auto()
OP_EQUAL = auto()
OP_LT = auto()
OP_GT = auto()
OP_IF = auto()
OP_END = auto()
OP_ELSE = auto()
OP_DUP = auto()
OP_WHILE = auto()
OP_DO = auto()
OP_MEM = auto()
OP_LOAD = auto()
OP_STORE = auto()
OP_COUNTER = auto()

TOKEN_NAMES = {
    OP_PRINT: 'print',
    OP_ADD: '+',
    OP_SUB: '-',
    OP_EQUAL: '==',
    OP_GT: '>',
    OP_LT: '<',
    OP_IF: 'if',
    OP_END: 'end',
    OP_ELSE: 'else',
    OP_DUP: 'dup',
    OP_WHILE: 'while',
    OP_DO: 'do',
    OP_MEM: 'mem',
    OP_STORE: '.',
    OP_LOAD: ',',
}

MEM_CAPACITY = 640_000


def simulate_program(program):
    stack = []
    assert OP_COUNTER == 16, 'Exhaustive handling of operators in simulation'
    i = 0
    while i < len(program):
        instruction = program[i]
        if instruction['type'] == OP_PUSH:
            stack.append(instruction['value'])
        elif instruction['type'] == OP_ADD:
            a = stack.pop()
            b = stack.pop()
            stack.append(a + b)
        elif instruction['type'] == OP_SUB:
            a = stack.pop()
            b = stack.pop()
            stack.append(b - a)
        elif instruction['type'] == OP_PRINT:
            a = stack.pop()
            print(a)
        elif instruction['type'] == OP_EQUAL:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(a == b))
        elif instruction['type'] == OP_LT:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(b < a))
        elif instruction['type'] == OP_GT:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(b > a))
        elif instruction['type'] == OP_IF:
            a = stack.pop()
            if a == 0:
                i = instruction['jmp']
        elif instruction['type'] == OP_ELSE:
            i = instruction['jmp']
        elif instruction['type'] == OP_END:
            if 'jmp' in instruction:
                i = instruction['jmp']
        elif instruction['type'] == OP_DUP:
            a = stack.pop()
            stack.append(a)
            stack.append(a)
        elif instruction['type'] == OP_WHILE:
            i += 1
            continue
        elif instruction['type'] == OP_DO:
            a = stack.pop()
            if a == 0:
                i = instruction['jmp']
        elif instruction['type'] == OP_MEM:
            raise NotImplementedError()
        elif instruction['type'] == OP_LOAD:
            raise NotImplementedError()
        elif instruction['type'] == OP_STORE:
            raise NotImplementedError()
        else:
            raise_error(f'Unhandled instruction: {TOKEN_NAMES[instruction["type"]]}',
                        instruction['loc'])
        i += 1


def compile_program(program):
    assert OP_COUNTER == 16, 'Exhaustive handling of operators in compilation'
    out = open('output.s', 'w')
    write_base = write_indent(out, 0)
    write_level1 = write_indent(out, 1)
    write_base('.section __DATA, __bss')
    write_base('mem:')
    write_level1(f'.skip {MEM_CAPACITY}')
    write_base('.section __TEXT, __text')
    write_base('.global _start')
    write_base('.align 3')
    asm_setup(write_base, write_level1)
    write_base('_start:')
    for i in range(len(program)):
        instruction = program[i]
        if instruction['type'] == OP_PUSH:
            write_level1(f'mov x0, #{instruction["value"]}')
            write_level1('push x0')
        elif instruction['type'] == OP_ADD:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('add x0, x0, x1')
            write_level1('push x0')
        elif instruction['type'] == OP_SUB:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('sub x0, x1, x0')
            write_level1('push x0')
        elif instruction['type'] == OP_PRINT:
            write_level1('pop x0')
            write_level1('bl dump')
        elif instruction['type'] == OP_EQUAL:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('cmp x0, x1')
            write_level1('cset x0, eq')
            write_level1('push x0')
        elif instruction['type'] == OP_LT:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('cmp x0, x1')
            write_level1('cset x0, gt')
            write_level1('push x0')
        elif instruction['type'] == OP_GT:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('cmp x0, x1')
            write_level1('cset x0, lt')
            write_level1('push x0')
        elif instruction['type'] in (OP_IF, OP_DO):
            write_level1('pop x0')
            write_level1('tst x0, x0')
            write_level1(f'b.eq end_{instruction["jmp"]}')
        elif instruction['type'] == OP_ELSE:
            write_level1(f'b end_{instruction["jmp"]}')
            write_base(f'end_{i}:')
        elif instruction['type'] == OP_END:
            if 'jmp' in instruction:
                write_level1(f'b while_{instruction["jmp"]}')
            write_base(f'end_{i}:')
        elif instruction['type'] == OP_DUP:
            write_level1('pop x0')
            write_level1('push x0')
            write_level1('push x0')
        elif instruction['type'] == OP_WHILE:
            write_base(f'while_{i}:')
        elif instruction['type'] == OP_MEM:
            write_level1('adrp x0, mem@PAGE')
            write_level1('push x0')
        elif instruction['type'] == OP_STORE:
            write_level1('popw w0')
            write_level1('pop x1')
            write_level1('strb w0, [x1]')
        elif instruction['type'] == OP_LOAD:
            write_level1('pop x0')
            write_level1('ldrb w1, [x0]')
            write_level1('pushw w1')
        else:
            raise_error(f'Unhandled instruction: {TOKEN_NAMES[instruction["type"]]}',
                        instruction['loc'])
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
    assert OP_COUNTER == 16, 'Exhaustive handling of tokens'
    token_dict = {
        'print': {'type': OP_PRINT, 'loc': location},
        '+': {'type': OP_ADD, 'loc': location},
        '-': {'type': OP_SUB, 'loc': location},
        '==': {'type': OP_EQUAL, 'loc': location},
        '>': {'type': OP_GT, 'loc': location},
        '<': {'type': OP_LT, 'loc': location},
        'if': {'type': OP_IF, 'loc': location},
        'end': {'type': OP_END, 'loc': location},
        'else': {'type': OP_ELSE, 'loc': location},
        'dup': {'type': OP_DUP, 'loc': location},
        'while': {'type': OP_WHILE, 'loc': location},
        'do': {'type': OP_DO, 'loc': location},
        'mem': {'type': OP_MEM, 'loc': location},
        '.': {'type': OP_STORE, 'loc': location},
        ',': {'type': OP_LOAD, 'loc': location},
    }
    if token in token_dict:
        return token_dict[token]
    elif token.isdigit() or (token[0] == '-' and token[1:].isdigit()):
        return {'type': OP_PUSH, 'loc': location, 'value': int(token)}
    else:
        raise_error(f'Unhandled token: {token}', location)


def cross_reference_blocks(program):
    assert OP_COUNTER == 16, 'Exhaustive handling of code block'
    stack = []
    for i in range(len(program)):
        if program[i]['type'] == OP_IF:
            stack.append(i)
        elif program[i]['type'] == OP_ELSE:
            if_index = stack.pop()
            if program[if_index]['type'] != OP_IF:
                raise_error(f'Else can only be used with an `if`', program[if_index]['loc'])
            program[if_index]['jmp'] = i
            stack.append(i)
        elif program[i]['type'] == OP_WHILE:
            stack.append(i)
        elif program[i]['type'] == OP_DO:
            stack.append(i)
        elif program[i]['type'] == OP_END:
            block_index = stack.pop()
            if program[block_index]['type'] in (OP_IF, OP_ELSE):
                program[block_index]['jmp'] = i
            elif program[block_index]['type'] == OP_DO:
                program[block_index]['jmp'] = i
                while_index = stack.pop()
                if program[while_index]['type'] != OP_WHILE:
                    raise_error('`while` must be present before `do`', program[while_index]['loc'])
                program[i]['jmp'] = while_index
            else:
                raise_error('End can only be used with an `if`, `else` or `while`',
                            program[block_index]['loc'])
    return program


def lex_line(line, file_path, line_number):
    l, r = 0, 0
    while l < len(line):
        if line[l].isspace():
            l += 1
            continue
        r = l
        while r < len(line) and not line[r].isspace():
            r += 1
        yield parse_token(line[l:r], (file_path, line_number, l))
        l = r


def lex_file(file_path):
    program = []
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
        subprocess.call('as -o output.o output.s', shell=True)
        subprocess.call(
            'ld -o output output.o -lSystem -syslibroot `xcrun -sdk macosx'
            ' --show-sdk-path` -e _start -arch arm64', shell=True)
        if '--run' in argv:
            subprocess.call('./output', shell=True)
    else:
        usage_help()
        exit(1)

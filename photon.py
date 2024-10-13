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


OP_PUSH_INT = auto(True)
OP_PUSH_STR = auto()
OP_ADD = auto()
OP_SUB = auto()
OP_PRINT = auto()
OP_EQUAL = auto()
OP_LT = auto()
OP_GT = auto()
OP_LTE = auto()
OP_GTE = auto()
OP_NE = auto()
OP_IF = auto()
OP_END = auto()
OP_ELSE = auto()
OP_DUP = auto()
OP_WHILE = auto()
OP_DO = auto()
OP_MEM = auto()
OP_LOAD = auto()
OP_STORE = auto()
OP_SYSCALL3 = auto()
OP_DUP2 = auto()
OP_DROP = auto()
OP_BITAND = auto()
OP_BITOR = auto()
OP_SHIFT_RIGHT = auto()
OP_SHIFT_LEFT = auto()
OP_SWAP = auto()
OP_OVER = auto()
OP_DROP2 = auto()
OP_MOD = auto()
OP_COUNTER = auto()

TOKEN_WORD = auto(True)
TOKEN_INT = auto()
TOKEN_STR = auto()
TOKEN_COUNTER = auto()

BUILTIN_NAMES = {
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
    OP_SYSCALL3: 'syscall3',
    OP_DUP2: 'dup2',
    OP_DROP: 'drop',
    OP_BITAND: '&',
    OP_BITOR: '|',
    OP_SHIFT_RIGHT: '<<',
    OP_SHIFT_LEFT: '>>',
    OP_SWAP: 'swap',
    OP_OVER: 'over',
    OP_DROP2: 'drop2',
    OP_MOD: '%',
    OP_GTE: '>=',
    OP_LTE: '<=',
    OP_NE: '!=',
}

assert OP_COUNTER == len(BUILTIN_NAMES) + 2, 'Exhaustive handling of built-in word names'

STR_CAPACITY = 640_000
MEM_CAPACITY = 640_000


def simulate_program(program):
    stack = []
    assert OP_COUNTER == 31, 'Exhaustive handling of operators in simulation'
    i = 0
    mem = bytearray(STR_CAPACITY + MEM_CAPACITY)
    str_size = 0
    while i < len(program):
        instruction = program[i]
        try:
            if instruction['type'] == OP_PUSH_INT:
                stack.append(instruction['value'])
            elif instruction['type'] == OP_PUSH_STR:
                bs = bytes(instruction['value'], 'utf-8')
                n = len(bs)
                stack.append(n)
                if 'addr' not in instruction:
                    instruction['addr'] = str_size
                    mem[str_size:str_size + n] = bs
                    str_size += n
                    if str_size > STR_CAPACITY:
                        raise_error('String buffer overflow', instruction['loc'])
                stack.append(instruction['addr'])
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
            elif instruction['type'] == OP_GTE:
                a = stack.pop()
                b = stack.pop()
                stack.append(int(b >= a))
            elif instruction['type'] == OP_LTE:
                a = stack.pop()
                b = stack.pop()
                stack.append(int(b <= a))
            elif instruction['type'] == OP_NE:
                a = stack.pop()
                b = stack.pop()
                stack.append(int(a != b))
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
            elif instruction['type'] == OP_DUP2:
                a = stack.pop()
                b = stack.pop()
                stack.append(b)
                stack.append(a)
                stack.append(b)
                stack.append(a)
            elif instruction['type'] == OP_DROP:
                stack.pop()
            elif instruction['type'] == OP_DROP2:
                stack.pop()
                stack.pop()
            elif instruction['type'] == OP_WHILE:
                i += 1
                continue
            elif instruction['type'] == OP_DO:
                a = stack.pop()
                if a == 0:
                    i = instruction['jmp']
            elif instruction['type'] == OP_MEM:
                stack.append(0)
            elif instruction['type'] == OP_LOAD:
                address = stack.pop()
                stack.append(mem[address])
            elif instruction['type'] == OP_STORE:
                value = stack.pop()
                address = stack.pop()
                mem[address] = value & 0xFF
            elif instruction['type'] == OP_BITOR:
                a = stack.pop()
                b = stack.pop()
                stack.append(a | b)
            elif instruction['type'] == OP_BITAND:
                a = stack.pop()
                b = stack.pop()
                stack.append(a & b)
            elif instruction['type'] == OP_SHIFT_RIGHT:
                a = stack.pop()
                b = stack.pop()
                stack.append(b >> a)
            elif instruction['type'] == OP_SHIFT_LEFT:
                a = stack.pop()
                b = stack.pop()
                stack.append(b << a)
            elif instruction['type'] == OP_SWAP:
                a = stack.pop()
                b = stack.pop()
                stack.append(a)
                stack.append(b)
            elif instruction['type'] == OP_OVER:
                a = stack.pop()
                b = stack.pop()
                stack.append(b)
                stack.append(a)
                stack.append(b)
            elif instruction['type'] == OP_MOD:
                a = stack.pop()
                b = stack.pop()
                stack.append(b % a)
            elif instruction['type'] == OP_SYSCALL3:
                syscall_number = stack.pop()
                arg1 = stack.pop()
                arg2 = stack.pop()
                arg3 = stack.pop()
                if syscall_number == 4:
                    if arg1 == 1:
                        print(mem[arg2:arg2 + arg3].decode(), end='')
                    elif arg1 == 2:
                        print(mem[arg2:arg2 + arg3].decode(), end='', file=sys.stderr)
                    else:
                        raise_error(f'Unknown file descriptor: {arg1}', instruction['loc'])
                else:
                    raise_error(f'Unknown syscall number: {syscall_number}', instruction['loc'])
            else:
                raise_error(f'Unhandled instruction: {BUILTIN_NAMES[instruction["type"]]}',
                            instruction['loc'])
        except Exception as e:
            raise_error(f'Exception in Simulation: {str(e)}', instruction['loc'])
        i += 1


def compile_program(program):
    assert OP_COUNTER == 31, 'Exhaustive handling of operators in compilation'
    out = open('output.s', 'w')
    write_base = write_indent(out, 0)
    write_level1 = write_indent(out, 1)
    write_base('.section __TEXT, __text')
    write_base('.global _start')
    write_base('.align 3')
    asm_setup(write_base, write_level1)
    write_base('_start:')
    strs = []
    allocated_strs = {}
    for i in range(len(program)):
        instruction = program[i]
        if instruction['type'] == OP_PUSH_INT:
            write_level1(f'ldr x0, ={instruction["value"]}')
            write_level1('push x0')
        elif instruction['type'] == OP_PUSH_STR:
            write_level1(f'ldr x0, ={len(instruction["value"])}')
            write_level1('push x0')
            write_level1(f'adrp x1, str_{allocated_strs.get(instruction["value"], len(strs))}@PAGE')
            write_level1('push x1')
            if instruction['value'] not in allocated_strs:
                allocated_strs[instruction['value']] = len(strs)
                strs.append(instruction['value'])
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
            write_level1('cmp x1, x0')
            write_level1('cset x0, lt')
            write_level1('push x0')
        elif instruction['type'] == OP_GT:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('cmp x1, x0')
            write_level1('cset x0, gt')
            write_level1('push x0')
        elif instruction['type'] == OP_LTE:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('cmp x1, x0')
            write_level1('cset x0, le')
            write_level1('push x0')
        elif instruction['type'] == OP_GTE:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('cmp x1, x0')
            write_level1('cset x0, ge')
            write_level1('push x0')
        elif instruction['type'] == OP_NE:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('cmp x0, x1')
            write_level1('cset x0, ne')
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
        elif instruction['type'] == OP_DUP2:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('push x1')
            write_level1('push x0')
            write_level1('push x1')
            write_level1('push x0')
        elif instruction['type'] == OP_DROP:
            write_level1('pop x0')
        elif instruction['type'] == OP_DROP2:
            write_level1('pop x0')
            write_level1('pop x0')
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
        elif instruction['type'] == OP_BITOR:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('orr x0, x0, x1')
            write_level1('push x0')
        elif instruction['type'] == OP_BITAND:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('and x0, x0, x1')
            write_level1('push x0')
        elif instruction['type'] == OP_SHIFT_RIGHT:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('lsr x0, x1, x0')
            write_level1('push x0')
        elif instruction['type'] == OP_SHIFT_LEFT:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('lsl x0, x1, x0')
            write_level1('push x0')
        elif instruction['type'] == OP_SWAP:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('push x0')
            write_level1('push x1')
        elif instruction['type'] == OP_OVER:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('push x1')
            write_level1('push x0')
            write_level1('push x1')
        elif instruction['type'] == OP_MOD:
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('udiv x2, x1, x0')
            write_level1('msub x0, x2, x0, x1')
            write_level1('push x0')
        elif instruction['type'] == OP_SYSCALL3:
            write_level1('pop x16')
            write_level1('pop x0')
            write_level1('pop x1')
            write_level1('pop x2')
            write_level1('svc #0')
        else:
            raise_error(f'Unhandled instruction: {BUILTIN_NAMES[instruction["type"]]}',
                        instruction['loc'])
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


def usage_help():
    print('Usage: photon.py <SUBCOMMAND> <FILENAME> <FLAGS>')
    print('Subcommands:')
    print('     sim     Simulate the program')
    print('     com     Compile the program')
    print('         --run   Used with `com` to run immediately')


def cross_reference_blocks(program):
    assert OP_COUNTER == 31, 'Exhaustive handling of code block'
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


def parse_token(token, location):
    assert OP_COUNTER == 31, 'Exhaustive handling of built-in words'
    builtin_words = {
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
        'syscall3': {'type': OP_SYSCALL3, 'loc': location},
        'dup2': {'type': OP_DUP2, 'loc': location},
        'drop': {'type': OP_DROP, 'loc': location},
        '&': {'type': OP_BITAND, 'loc': location},
        '|': {'type': OP_BITOR, 'loc': location},
        '>>': {'type': OP_SHIFT_RIGHT, 'loc': location},
        '<<': {'type': OP_SHIFT_LEFT, 'loc': location},
        'swap': {'type': OP_SWAP, 'loc': location},
        'over': {'type': OP_OVER, 'loc': location},
        'drop2': {'type': OP_DROP2, 'loc': location},
        '%': {'type': OP_MOD, 'loc': location},
        '>=': {'type': OP_GTE, 'loc': location},
        '<=': {'type': OP_LTE, 'loc': location},
        '!=': {'type': OP_NE, 'loc': location},
    }
    assert TOKEN_COUNTER == 3, "Exhaustive handling of tokens"
    if token['type'] == TOKEN_WORD:
        return builtin_words[token['value']]
    elif token['type'] == TOKEN_INT:
        return {'type': OP_PUSH_INT, 'loc': location, 'value': token['value']}
    elif token['type'] == TOKEN_STR:
        return {'type': OP_PUSH_STR, 'loc': location, 'value': token['value']}
    else:
        raise_error(f'Unhandled token: {token}', location)


def seek_until(line, start, predicate):
    while start < len(line) and not predicate(line[start]):
        start += 1
    return start


def lex_line(line, file_path, line_number):
    col = seek_until(line, 0, lambda x: not x.isspace())
    while col < len(line):
        location = (file_path, line_number, col)
        if line[col] == '"':
            col_end = seek_until(line, col + 1, lambda x: x == '"')
            word = line[col + 1:col_end]
            yield parse_token({'type': TOKEN_STR, 'value': word.encode('utf-8').decode('unicode_escape')},
                              location)
            col = seek_until(line, col_end + 1, lambda x: not x.isspace())
        else:
            col_end = seek_until(line, col, lambda x: x.isspace())
            word = line[col:col_end]
            try:
                yield parse_token({'type': TOKEN_INT, 'value': int(word)},
                                  location)
            except ValueError:
                yield parse_token({'type': TOKEN_WORD, 'value': word},
                                  location)
            col = seek_until(line, col_end, lambda x: not x.isspace())


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

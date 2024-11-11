# Photon

Photon is a stack-based programming language implemented
in Python.

## Features

- [x] Native
- [x] Executable
- [x] Stack-based
- [x] [Turing Complete (Shown by Rule 110 Implementation)](./examples/rule110.phtn)
- [x] Static Type Checking

Planned:
- [ ] Self-Hosted ([./photon.phtn](./photon.phtn) for current progress)

## Examples

Hello World:
``` pascal
include "std.phtn"

"Hello World!\n" puts
```

Printing numbers from 0 to 99:
``` pascal
0 while 100 < do
    dup print
end drop
```

## Quick Start

There are 2 ways to execute a `./phtn` file:

- Simulation
- Compilation

### Simulation

This mode interprets the program emulating a MacOS Little
Endian environment

``` bash
$ python3 photon.py sim program.phtn
```

_**Note:**_ It is recommended to use PyPy for the Simulation
mode as CPython is too slow for that.

### Compilation

This mode generates an armv8 assembly file, compiles it with
`as`, and then statically links with `ld`.

``` bash
$ python3 photon.py com program.phtn
```

### Tests

When executed the `test.py` script will run all Photon files
in the examples folder with both `sim` and `com` subcommands
and compare their outputs to the ones saved in `./examples/examples_output.test`

#### Tests Usage

```
Usage: python3 test.py [<FLAGS>]
Flags:
    --snapshot                  Record the output of ./examples/*
    [-v | --verbose]            Show `sim` and `com` outputs for each file
    --record-argv <filename>    Record command line arguments for a file
    --record-stdin <filename>   Record standard input for a file
```

### Photon Usage

```
Usage: photon.py <SUBCOMMAND> <FILENAME> [<FLAGS>]
Subcommands:
     sim     Simulate the program in a macos little endian environment
     flow    Generate a control flow graph for the given program
     com     Compile the program to arm 64-bit assembly
Flags:
         [-d | --debug]  Use the compiler in debug mode
         --unsafe        Disable type checking
         --run           Used with `com` to run immediately
```

## Language Reference

### Literals

#### Integer

When the compiler encounters an integer it pushes that integer onto the stack

`24`

#### String

The strings support escaping using `unicode_escape` of Python.
When the compilers encounters a string of the form `"..."`:

1. It pushes the size of the string in bytes
2. It copies the contents of the string into memory
3. It pushes the pointer to the start of the string in memory

`"Hello\n"`

#### C-String

When a string literal is of the form `"...\0"`
(ends with null terminator) the compiler does not push the
size of the string, and the string is null terminated in memory

`"Hello\0"`

#### Character

When the compiler encounter a single character encolsed in `'` quotes,
it pushes the ASCII value corresponding to that character onto
the stack

`'a'`

### Intrinsics (Buil-in Words)


#### Stack Operations

| Name    | Signature        | Description                                 |
|---------|------------------|---------------------------------------------|
| `dup`   | `a -> a a`       | duplicate the element on top of the stack.  |
| `swap`  | `a b -> b a`     | swap 2 elements on the top of the stack.    |
| `drop`  | `a b -> a`       | drops the top element of the stack.         |
| `print` | `a b -> a`       | pop the top element and print into stdout.  |
| `over`  | `a b -> a b a`   | copy the element below the top of the stack |
| `rot`   | `a b c -> b c a` | rotate the top three stack elements.        |

#### Comparison Operations


| Name | Signature                          | Description                                                                             |
|------|------------------------------------|-----------------------------------------------------------------------------------------|
| `= ` | `[a: T] [b: T] -> [a == b : bool]` | checks if the two elements on top of the stack are equal, where T is `int`,`ptr` or `bool`. |
| `!=` | `[a: T] [b: T] -> [a != b : bool]` | checks if the two elements on top of the stack are not equal, where T is `int`,`ptr` or `bool`. |
| `> ` | `[a: T] [b: T] -> [a > b  : bool]` | applies the greater than comparison on top two elements, where T is `int` or `ptr`.         |
| `< ` | `[a: T] [b: T] -> [a < b  : bool]` | applies the less than comparison on top two elements, where T is `int` or `ptr`.            |
| `>=` | `[a: T] [b: T] -> [a >= b : bool]` | applies the greater than or equal comparison on top two elements, where T is `int` or `ptr`. |
| `<=` | `[a: T] [b: T] -> [a <= b : bool]` | applies the less than or equal comparison on top two elements, where T is `int` or `ptr`.   |

#### Arithmetic

| Name | Signature                           | Description                                                                                                                                         |
|------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `+`  | `[a: T] [b: U] -> [a + b: V]`       | sums up the two elements on the top of the stack, where the possible triples `T U -> V` are `int int -> int`, `int ptr -> ptr`, `ptr int -> ptr`.   |
| `-`  | `[a: T] [b: U] -> [a - b: V]`       | subtracts the two elements on the top of the stack, where the possible triples `T U -> V` are `int int -> int`, `ptr int -> ptr`, `ptr ptr -> int`. |
| `*`  | `[a: int] [b: int] -> [a * b: int]` | multiples the two elements on top of the stack                                                                                                      |
| `/`  | `[a: int] [b: int] -> [a / b: int]` | divides the two elements on top of the stack                                                                                                        |

#### Bitwise

| Name   | Signature                            | Description                                                                   |
|--------|--------------------------------------|-------------------------------------------------------------------------------|
| `>>`   | `[a: int] [b: int] -> [a >> b: int]` | right **unsigned** bit shift.                                                 |
| `<<`   | `[a: int] [b: int] -> [a << b: int]` | left bit shift.                                                               |
| `or`   | `[a: T] [b: T] -> [a \| b: T]`       | bitwise `or`, which also acts as logical `or`, where `T` is `int` or `bool`   |
| `and`  | `[a: T] [b: T] -> [a & b: T]`        | bitwise `and`, which also acts as logical `and`, where `T` is `int` or `bool` |
| `bnot` | `[a: int] -> [~a: int]`              | bitwise `not`.                                                                |

**_Note:_** As there are no boolean literals in Photon, `1` and `0`
are used to represent booleans (that's why `bnot` will
not behave as logical `not`, there is a logical 
`not` implementation in `std.phtn`)

#### Memory

| Name     | Signature                    | Description                                                                  |
|----------|------------------------------|------------------------------------------------------------------------------|
| `!`      | `[byte: int] [addr: ptr] ->` | store a given byte at the address on the stack.                              |
| `@`      | `[addr: ptr] -> [byte: int]` | load a byte from the address on the stack.                                   |
| `!2`     | `[byte: int] [addr: ptr] ->` | store a 2-byte word at the address on the stack.                             |
| `@2`     | `[addr: ptr] -> [byte: int]` | load a 2-byte word from the address on the stack.                            |
| `!4`     | `[byte: int] [addr: ptr] ->` | store a 4-byte word at the address on the stack.                             |
| `@4`     | `[addr: ptr] -> [byte: int]` | load a 4-byte word from the address on the stack.                            |
| `!8`     | `[byte: int] [addr: ptr] ->` | store an 8-byte word at the address on the stack.                            |
| `@8`     | `[addr: ptr] -> [byte: int]` | load an 8-byte word from the address on the stack.                           |
| `->int`  | `[a: T] -> [a: int]`         | cast the element on top of the stack to `int`, where `T` is `bool` or `int`  |
| `->bool` | `[a: T] -> [a: bool]`        | cast the element on top of the stack to `bool`, where `T` is `bool` or `int` |
| `->ptr`  | `[a: T] -> [a: ptr]`         | cast the element on top of the stack to `ptr`, where `T` is `ptr` or `int`   |

#### System

- `syscall<n>` - perform a syscall with n arguments where n is in the range `[0..6]`. (`syscall1`, `syscall2`, etc)
```
arg1 ... argn SYSCALL_NUMBER syscall<n> -> [return_code: int]
```

#### Misc

- `here [ -> [len: int] [str: ptr]]` - pushes a string `"<file-path>:<row>:<col>"` where `<file-path>` is the path to the file where `here` is located, `<row>` is the row on which `here` is located and `<col>` is the column from which `here` starts.
- `argc [ -> [argc: int]]` Pushes command line argument count onto stack
- `argv [ -> [argv: ptr]]` Pushes a pointer to the start of `argv` (which on MacOS is an array of pointers to the arguments)

### Control Flow

- `if <condition> do <then-branch> else <else-branch> end` - pops the element on top of the stack and if the element is not `0` executes the `<then-branch>`, otherwise `<else-branch>`.
- `while <condition> do <body> end` - keeps executing both `<condition>` and `<body>` until `<condition>` produces `0` at the top of the stack. Checking the result of the `<condition>` removes it from the stack.
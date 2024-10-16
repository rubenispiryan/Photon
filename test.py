import os
import re
import subprocess
import sys
from typing import List

TEST_FILE_NAME = 'examples/examples_output.test'


def execute(subcommand: str, filename: str, flag: str ='') -> str:
    try:
        return subprocess.check_output(f'pypy3.10 photon.py {subcommand} {filename} {flag}', shell=True,
                                       stderr=subprocess.STDOUT).decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f'Command: "pypy3.10 photon.py {subcommand} {filename} {flag}" failed with error:')
        print(e.output.decode('utf-8'))
        exit(1)


def get_files(folder: str) -> List[str]:
    output = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isdir(file_path):
            output.extend(get_files(file_path))
        else:
            output.append(file_path)
    return output


def read_expected(filename: str) -> str | None:
    with open(TEST_FILE_NAME, 'r') as f:
        pattern = rf'{filename}\n(.*?)\nend{filename}\n'

        match = re.search(pattern, f.read(), re.DOTALL)

        if match:
            return match.group(1)
        else:
            return None

def create_expected_file(filename: str) -> None:
    with open(filename, 'w') as f:
        f.write('')

def add_expected_output(filename: str, expected: str) -> None:
    with open(filename, 'a') as f:
        f.write(expected)


def create_examples() -> None:
    filenames = filter(lambda x: x.endswith('.phtn'), get_files('./examples'))
    for filename in filenames:
        simulated_out = execute('sim', filename)
        compiled_out = execute('com', filename, flag='--run')
        assert compiled_out == simulated_out, (
            f'Output from compilation:\n'
            f'  {compiled_out!r}\n'
            f'Output from simulation:\n'
            f'  {simulated_out!r}\n'
            f'Compilation and Simulation of {filename} do not match during snapshot')
        snapshot_output = f'{filename}\n{compiled_out}\nend{filename}\n'
        add_expected_output(TEST_FILE_NAME, snapshot_output)
        print(f'Snapshot of {filename} added successfully')


def check_examples() -> None:
    filenames = filter(lambda x: x.endswith('.phtn'), get_files('./examples'))
    for filename in filenames:
        if (expected := read_expected(filename)) is None:
            print(f'Tests for {filename} do not exist')
            continue
        simulated_out = execute('sim', filename)
        compiled_out = execute('com', filename, flag='--run')
        assert compiled_out == expected, (
            f'Output from compilation:\n'
            f'  {compiled_out!r}\n'
            f'Expected:\n'
            f'  {expected!r}\n'
            f'Compilation and Test of {filename} do not match')
        assert simulated_out == expected, (
            f'Output from simulation:\n'
            f'{simulated_out!r}\n'
            f'Expect:\n'
            f'{expected!r}\n'
            f'Simulation and Test of {filename} do not match')
        print(f'Test of {filename} passed successfully')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_examples()
    elif len(sys.argv) == 2 and sys.argv[1] == '--snapshot':
        create_expected_file(TEST_FILE_NAME)
        create_examples()

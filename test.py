import os
import re
import subprocess
import sys
from typing import List

TEST_FILE_NAME = 'examples/examples_output.test'
SEPARATOR = '-#-' * 20


def process_exception_message(message: str) -> str:
    output = ''
    for line in message.splitlines():
        output += re.split('\[(ERROR|NOTE)\] ', line)[-1] + '\n'
    return output


def execute(subcommand: str, filename: str, flags: str = '') -> str:
    try:
        return subprocess.check_output(f'pypy3.10 photon.py {subcommand} {filename} {flags}', shell=True,
                                       stderr=subprocess.STDOUT).decode('utf-8')
    except subprocess.CalledProcessError as e:
        output = 'STDOUT:\n' + process_exception_message(e.output.decode('utf-8'))
        return output + 'Exit code: ' + str(e.returncode)


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
        pattern = rf'{filename}\n(.*?)\n{SEPARATOR}\n'

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
        compiled_out = execute('com', filename, flags='--run')
        assert compiled_out == simulated_out, (
            f'Output from compilation:\n'
            f'  {compiled_out!r}\n'
            f'Output from simulation:\n'
            f'  {simulated_out!r}\n'
            f'Compilation and Simulation of {filename} do not match during snapshot')
        snapshot_output = f'{filename}\n{compiled_out}\n{SEPARATOR}\n'
        add_expected_output(TEST_FILE_NAME, snapshot_output)
        print(f'Snapshot of {filename} added successfully')


def check_examples(debug: bool = False) -> dict[str, str]:
    filenames = filter(lambda x: x.endswith('.phtn'), get_files('./examples'))
    summary = {}
    for filename in filenames:
        if (expected := read_expected(filename)) is None:
            print(f'Tests for {filename} do not exist')
            summary[filename] = 'UNDEFINED'
            continue
        simulated_out = execute('sim', filename)
        compiled_out = execute('com', filename, flags='--run')
        if compiled_out != expected:
            if debug:
                print(
                    f'Expected:\n'
                    f'  {expected}\n'
                    f'Output from compilation:\n'
                    f'  {compiled_out}\n', file=sys.stderr)
            print(f'Compilation and Test of {filename} do not match', file=sys.stderr)
            summary[filename] = 'COM:\n' + compiled_out
        elif simulated_out != expected:
            if debug:
                print(
                    f'Expect:\n'
                    f'{expected}\n'
                    f'Output from simulation:\n'
                    f'{simulated_out}\n', file=sys.stderr)
            print(f'Simulation and Test of {filename} do not match', file=sys.stderr)
            summary[filename] = 'SIM:\n' + simulated_out
        else:
            print(f'Test of {filename} passed successfully')
            summary[filename] = ''
    return summary


def print_summary(summary: dict[str, str]) -> None:
    count_passed = 0
    undefined_count = 0
    failed_outputs = ''
    for filename in summary:
        if not summary[filename]:
            count_passed += 1
        elif summary[filename] == 'UNDEFINED':
            undefined_count += 1
        else:
            failed_outputs += f'{filename}:\n{summary[filename]}'
    print(f'Total tests: {len(summary)}, Passed: {count_passed},'
          f' Failed: {len(summary) - count_passed - undefined_count}, Undefined: {undefined_count}')
    if failed_outputs:
        print(f'Failed tests:\n{failed_outputs}')



if __name__ == '__main__':
    if len(sys.argv) == 1:
        print_summary(check_examples())
    elif len(sys.argv) == 2 and sys.argv[1] == '--snapshot':
        create_expected_file(TEST_FILE_NAME)
        create_examples()

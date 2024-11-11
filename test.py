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


def execute(subcommand: str, filename: str, flags: str = '', argv: str = '', stdin: str ='') -> str:
    command = f'pypy3.10 photon.py {subcommand} {filename} {flags} {argv}'
    result = subprocess.run(command, input=stdin.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=True)
    if result.returncode != 0:
        full_output = result.stderr.decode('utf-8') + result.stdout.decode('utf-8')
        output = process_exception_message(full_output)
        return output + 'Exit code: ' + str(result.returncode)
    return result.stdout.decode('utf-8')



def get_files(folder: str) -> List[str]:
    output = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isdir(file_path):
            output.extend(get_files(file_path))
        else:
            output.append(file_path)
    return output


def read_expected(filename: str) -> dict[str, str] | None:
    with open(TEST_FILE_NAME, 'r') as f:
        pattern = rf'{filename} argv (.*?) endargv\nSTDIN:\n(.*?)\nSTDOUT:\n(.*?)\nend_{filename}{SEPARATOR}\n'
        match = re.search(pattern, f.read(), re.DOTALL)

        if match:
            argv_match = match.group(1)
            stdin_match = match.group(2)
            expected_text = match.group(3)
            return {'expected_text': expected_text, 'stdin': stdin_match, 'argv': argv_match}
        else:
            return None


def create_expected_file(filename: str) -> None:
    with open(filename, 'w') as f:
        f.write('')


def add_expected_output(filename: str, expected: str) -> None:
    with open(filename, 'a') as f:
        f.write(expected)

def replace_expected_output(filename: str, expected: str) -> None:
    with open(TEST_FILE_NAME, 'r+') as f:
        pattern = rf'{filename} argv (.*?) endargv\nSTDIN:\n(.*?)\nSTDOUT:\n(.*?)\nend_{filename}{SEPARATOR}\n'
        text = f.read()
        replaced_tests = re.sub(pattern, expected, text, flags=re.DOTALL)
        f.seek(0)
        f.write(replaced_tests)

def create_example(filename: str, argv: str = '', stdin: str = '') -> None:
    expected_data = read_expected(filename)
    if expected_data and argv == '':
        argv = expected_data['argv']
    if expected_data and stdin == '':
        stdin = expected_data['stdin']
    simulated_out = execute('sim', filename, argv=argv, stdin=stdin)
    compiled_out = execute('com', filename, flags='--run', argv=argv, stdin=stdin)
    assert compiled_out == simulated_out, (
        f'Output from compilation:\n'
        f'  {compiled_out}\n'
        f'Output from simulation:\n'
        f'  {simulated_out}\n'
        f'Compilation and Simulation of {filename} do not match during snapshot')
    snapshot_output = f'{filename} argv {argv} endargv\nSTDIN:\n{stdin}\nSTDOUT:\n{compiled_out}\nend_{filename}{SEPARATOR}\n'
    if expected_data:
        replace_expected_output(filename, snapshot_output)
    else:
        add_expected_output(TEST_FILE_NAME, snapshot_output)
    print(f'Snapshot of {filename} added successfully')

def create_examples() -> None:
    filenames = filter(lambda x: x.endswith('.phtn'), get_files('./examples'))
    for filename in filenames:
        create_example(filename)


def check_examples(verbose: bool = False) -> dict[str, str]:
    filenames = filter(lambda x: x.endswith('.phtn'), get_files('./examples'))
    summary = {}
    for filename in filenames:
        if (test_data := read_expected(filename)) is None:
            print(f'Test for {filename} does not exist')
            summary[filename] = 'UNDEFINED'
            continue
        expected = test_data['expected_text']
        input_args = test_data['argv']
        stdin = test_data['stdin']
        simulated_out = execute('sim', filename, argv=input_args, stdin=stdin)
        compiled_out = execute('com', filename, flags='--run', argv=input_args, stdin=stdin)
        if compiled_out != expected:
            if verbose:
                print(
                    f'Expected:\n'
                    f'  {expected}\n'
                    f'Output from compilation:\n'
                    f'  {compiled_out}\n', file=sys.stderr)
            print(f'Compilation and Test of {filename} do not match', file=sys.stderr)
            summary[filename] = 'COM:\n' + compiled_out
        elif simulated_out != expected:
            if verbose:
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
            failed_outputs += f'{filename}:\n{summary[filename]}\n'
    if failed_outputs:
        print(f'Failed tests:\n{failed_outputs}')
    print(f'Total tests: {len(summary)}, Passed: {count_passed},'
          f' Failed: {len(summary) - count_passed - undefined_count}, Undefined: {undefined_count}')


def record_argv(arg_filenames: list[str]) -> None:
    filenames = set(filter(lambda x: x.endswith('.phtn'), get_files('./examples')))
    for arg_filename in arg_filenames:
        if arg_filename in filenames:
            argv = input('Enter argv: ')
            create_example(arg_filename, argv=argv)


def record_stdin(arg_filenames: list[str]) -> None:
    filenames = set(filter(lambda x: x.endswith('.phtn'), get_files('./examples')))
    for arg_filename in arg_filenames:
        if arg_filename in filenames:
            print("Enter stdin (press Ctrl+D to finish):")
            stdin = sys.stdin.read()
            create_example(arg_filename, stdin=stdin)

# TODO: Some examples cannot be simulated
if __name__ == '__main__':
    verb = '-v' in sys.argv or '--verbose' in sys.argv
    reset = '--reset' in sys.argv
    if len(sys.argv) == 1 or len(sys.argv) == 2 and verb:
        print_summary(check_examples(verb))
    elif len(sys.argv) > 1 and sys.argv[1] == '--snapshot':
        if not os.path.exists(TEST_FILE_NAME) or reset:
            create_expected_file(TEST_FILE_NAME)
        create_examples()
    elif len(sys.argv) > 2 and sys.argv[1] == '--record-argv':
        record_argv(sys.argv[2:])
    elif len(sys.argv) > 2 and sys.argv[1] == '--record-stdin':
        record_stdin(sys.argv[2:])
    else:
        print(f'Usage: {sys.argv[0]} [--snapshot] [-d | --debug] [--record-(argv|stdin) <filename>]')

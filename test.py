import os
import subprocess
import sys


def execute(subcommand, filename, flag=''):
    return subprocess.check_output(f'python3 photon.py {subcommand} {filename} {flag}', shell=True)


def get_files(folder):
    output = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isdir(file_path):
            get_files(file_path)
        else:
            output.append(file_path)
    return output


def read_expected(filename):
    with open(filename, 'rb') as f:
        expected = f.read()
        return expected

def create_expected(filename, expected):
    with open(filename, 'wb') as f:
        f.write(expected)


def create_examples():
    filenames = filter(lambda x: x.endswith('.phtn'), get_files('./examples'))
    for filename in filenames:
        simulated_out = execute('sim', filename)
        compiled_out = execute('com', filename, flag='--run')
        assert compiled_out == simulated_out, (
            f'Output from compilation:\n'
            f'  {compiled_out}\n'
            f'Output from simulation:\n'
            f'  {simulated_out}\n'
            f'Compilation and Simulation of {filename} do not match during snapshot')
        create_expected(filename[:-4] + 'test', compiled_out)
        print(f'Snapshot of {filename} created successfully')


def check_examples():
    filenames = filter(lambda x: x.endswith('.phtn'), get_files('./examples'))
    for filename in filenames:
        expected_filename = filename[:-4] + 'test'
        if not os.path.exists(expected_filename):
            print(f'Tests for {filename} do not exist')
            continue
        simulated_out = execute('sim', filename)
        compiled_out = execute('com', filename, flag='--run')
        expected = read_expected(filename[:-4] + 'test')
        assert compiled_out == expected, (
            f'Output from compilation:\n'
            f'  {compiled_out}\n'
            f'Expected:\n'
            f'  {expected}\n'
            f'Compilation and Test of {filename} do not match')
        assert simulated_out == expected, (
            f'Output from simulation:\n'
            f'{simulated_out}\n'
            f'Expect:\n'
            f'{expected}\n'
            f'Simulation and Test of {filename} do not match')
        print(f'Test of {filename} passed successfully')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_examples()
    elif len(sys.argv) == 2 and sys.argv[1] == '--snapshot':
        create_examples()

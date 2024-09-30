import os
import subprocess


def execute(subcommand, filename):
    return subprocess.check_output(f'python3 photon.py {subcommand} {filename}', shell=True)


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


def main():
    filenames = filter(lambda x: x.endswith('.phtn'), get_files('./examples'))
    for filename in filenames:
        simulated_out = execute('sim', filename)
        compiled_out = execute('com', filename)
        expected = read_expected(filename[:-4] + 'test')
        assert compiled_out == expected, f'Compilation and Test of {filename} do not match'
        assert simulated_out == expected, f'Simulation and Test of {filename} do not match'


if __name__ == '__main__':
    main()

import os
import subprocess


if __name__ == '__main__':
    path = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/Whitening_Agent_Black_Box/results/debug'
    number_data = 5000

    current_directory = os.getcwd()
    current_directory = os.path.join(current_directory, '..')

    if os.path.isdir(current_directory):
        command = f'bokeh serve --show {current_directory} --args {path} {number_data}'
        subprocess.run(command, shell=True)
    else:
        print('The directory was not found.')

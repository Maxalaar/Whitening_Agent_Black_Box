import os
import subprocess


if __name__ == '__main__':
    current_directory = os.getcwd()
    current_directory = os.path.join(current_directory, '..')

    if os.path.isdir(current_directory):
        command = f'bokeh serve --show {current_directory}'
        subprocess.run(command, shell=True)
    else:
        print('The directory was not found.')

import platform
import subprocess
from pathlib import Path

from fraud_detection.utils import MLFLOW_DIR

CMD_SEP = '&' if platform.system() == 'Windows' else ';'
SELF_PATH = Path(__file__).parent.absolute()
NBS_PATH = SELF_PATH / 'notebooks'
DOIT_CONFIG = {'default_tasks': ['nodes', 'format', 'formatnb']}


def syscmd(string):
    subprocess.call(string, shell=True)
    return True


def task_format():
    """makes code organized and pretty"""
    nparts = len(SELF_PATH.parts)
    for filepath in SELF_PATH.glob('**/*.py'):
        if str(filepath).startswith(str(SELF_PATH / 'output')):
            continue
        yield {
            'name':
            '/'.join(filepath.parts[nparts:]),
            'actions':
            [('autoflake -i -r --expand-star-imports' +
              ' --remove-all-unused-imports' +
              ' --remove-duplicate-keys --remove-unused-variables %s' +
              ' %s isort %s %s yapf -i -r %s') %
             (filepath, CMD_SEP, filepath, CMD_SEP, filepath)],
            'file_dep': [filepath],
            'verbosity':
            2
        }


def task_formatnb():
    """makes notebooks organized and pretty"""
    nparts = len(NBS_PATH.parts)
    for filepath in NBS_PATH.glob('*.ipynb'):
        filename = filepath.as_posix()
        yield {
            'name':
            '/'.join(filepath.parts[nparts:]),
            'actions': [('nbqa isort "%s" %s nbqa yapf -i -r "%s"') %
                        (filename, CMD_SEP, filename)],
            'file_dep': [filepath],
            'verbosity':
            2
        }


def task_pytest():
    """run pytests under tests folder"""
    return {
        'actions': [lambda: syscmd('pytest tests/ --disable-pytest-warnings')],
        'verbosity': 2
    }


def task_ui():
    """start mlflow ui"""
    return {
        'actions':
        ['mlflow ui --backend-store-uri file:%s' % MLFLOW_DIR.as_posix()],
        'verbosity':
        2
    }


def task_nodes():
    """fix nodes __init__ imports"""
    NODES_PATH = Path(__file__).parent / 'fraud_detection/nodes'
    paths = [p for p in NODES_PATH.glob('*.py') if p.stem != '__init__']
    modules = [p.stem for p in paths]

    def nodes():

        new_init_content = ''

        for module in modules:
            new_init_content += f'from .{module} import {module}\n'

        new_init_content += '\n__all__ = ['
        for module in modules:
            new_init_content += f"'{module}', "

        new_init_content = new_init_content[:-2]
        new_init_content += ']\n'
        print(new_init_content)

        with open(NODES_PATH / '__init__.py', 'w') as f:
            f.write(new_init_content)

    return {'actions': [nodes], 'file_dep': paths}

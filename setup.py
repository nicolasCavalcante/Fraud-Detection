from pathlib import Path

from setuptools import find_packages, setup

SELF_PATH = Path(__file__).parent.absolute()


def read(path: Path):
    with open(path, 'r') as f:
        return f.read()


setup(
    name='fraud_detection',
    description='Preserve bank money by detecting fraudulent transactions from transactions history',
    author="Nicolas Garcia Cavalcante",
    author_email='nicolasgcavalcante@gmail.com',
    packages=find_packages(include=['fraud_detection', 'fraud_detection.*']),
    include_package_data=True,
    license="MIT license",
    keywords='fraud_detection',
    url='https://github.com/nicolasCavalcante/fraud_detection',
    long_description=read(SELF_PATH / 'README.md'),
)

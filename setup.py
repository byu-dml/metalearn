from setuptools import setup, find_packages

__version__ = '0.5.4'

setup(
    name = 'metalearn',
    packages = find_packages(include=['metalearn', 'metalearn.*']),
    version = __version__,
    description = 'A package to aid in metalearning',
    author = 'Roland Laboulaye, Brandon Schoenfeld, Casey Davis',
    author_email = 'rlaboulaye@gmail.com, bjschoenfeld@gmail.com, caseykdavis@gmail.com',
    url = 'https://github.com/byu-dml/metalearn',
    download_url = 'https://github.com/byu-dml/metalearn/archive/{}.tar.gz'.format(__version__),
    keywords = ['metalearning', 'machine learning', 'metalearn'],
    install_requires = [
        'numpy<=1.15.4',
        'scikit-learn<=0.20.3',
        'pandas<=0.23.4'
    ],
    classifiers = [
        'Programming Language :: Python :: 3.6'
    ],
    python_requires='~=3.6',
    include_package_data=True,
)

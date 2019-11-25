from setuptools import setup, find_packages

__version__ = '0.6.0'

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
        'numpy<=1.17.3',
        'scikit-learn<=0.21.3',
        'pandas<=0.25.2'
    ],
    classifiers = [
        'Programming Language :: Python :: 3.6'
    ],
    python_requires='~=3.6',
    include_package_data=True,
)

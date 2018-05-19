from setuptools import setup, find_packages

__version__ = '0.4.5'

setup(
    name = 'metalearn',
    packages = find_packages(),
    version = __version__,
    description = 'A package to aid in metalearning',
    author = 'Roland Laboulaye, Brandon Schoenfeld, Casey Davis',
    author_email = 'rlaboulaye@gmail.com, bjschoenfeld@gmail.com, caseykdavis@gmail.com',
    url = 'https://github.com/byu-dml/metalearn',
    download_url = 'https://github.com/byu-dml/metalearn/archive/{}.tar.gz'.format(__version__),
    keywords = ['metalearning', 'machine learning', 'metalearn'],
    install_requires = [
        'numpy==1.14.2',
        'pandas==0.22.0',
        'python-dateutil==2.7.2',
        'pytz==2018.4',
        'scikit-learn==0.19.1',
        'scipy==1.0.1',
        'six==1.11.0'
    ],
    classifiers = [
        'Programming Language :: Python :: 3.6'
    ],
    python_requires='~=3.6',
    include_package_data=True,
    tests_require=[
        'arff2pandas==1.0.1',
        'liac-arff==2.2.1'
    ]
)

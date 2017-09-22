from setuptools import setup, find_packages
setup(
  name = 'metalearn',
  packages = find_packages(),
  version = '0.1.7',
  description = 'A package to retrieve metafeatures from datasets',
  author = 'Casey Davis',
  author_email = 'caseykdavis@gmail.com',
  url = 'https://github.com/byu-dml/metalearn',
  download_url = 'https://github.com/byu-dml/metalearn/archive/0.1.7.tar.gz',
  keywords = ['metalearning', 'machine learning', 'darpa', 'metalearn'],
  entry_points = {
    'd3m.primitives' : [
      'metalearn.MetaFeatures = metalearn.features.metafeatures:MetaFeatures',
    ],
  },
  install_requires = ["numpy", "sklearn", "pandas", "scipy", "h5py", "git+ssh://git@gitlab.datadrivendiscovery.org/d3m/primitive-interfaces.git"],
  classifiers = [],
)
from setuptools import setup, find_packages
setup(
  name = "metalearn",
  packages = find_packages(),
  version = "0.2.0",
  description = "A package to aid in metalearning",
  author = "Casey Davis, Roland Laboulaye, Brandon Schoenfeld",
  author_email = "caseykdavis@gmail.com, rlaboulaye@gmail.com, bjschoenfeld@gmail.com",
  url = "https://github.com/byu-dml/metalearn",
  download_url = "https://github.com/byu-dml/metalearn/archive/0.2.0.tar.gz",
  keywords = ["metalearning", "machine learning", "metalearn"],
  install_requires = ["numpy", "sklearn", "pandas", "scipy", "h5py"],
  classifiers = [],
)

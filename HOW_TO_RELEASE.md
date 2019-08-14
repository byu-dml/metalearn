# How To Release the `metalearn` Package

### Testing

1. Using a clean environment, do a final test of the current state of the `develop` branch. Make sure before tests are run to delete or move the `metalearn` directory so the tests are run against the installed version of `metalearn` and not the local files.
1. Test using the develop branch of the `metalearn` package with our [`d3m_primitives`](https://github.com/byu-dml/d3m-primitives) package, to make sure the APIs are compatible.

### Updating the Version

1. If all the tests pass, bump up the version number for `metalearn` inside `setup.py` of the `develop` branch.
1. Merge the up-to-date `develop` into the up-to-date `master` locally, then run the tests one more time to ensure the merge happened successfully.
1. Push the merged `master` to remote.
1. If the Travis CI build passes, go to 'Releases', and create a new release with the correct version number for the master branch.

### Publishing to PyPi

1. Run `python3 setup.py sdist bdist_wheel` from the clean, new version of `master` to build the newly-released version of the package.
1. Run `pip3 install twine`
1. Run `twine upload dist/*` to upload the whole package distribution to PyPi. You will need to authenticate with PyPi credentials connected to the `metalearn` package.
1. In a new directory, with a new python environment, run `pip install metalearn` and verify the version is the new version number you've just created.
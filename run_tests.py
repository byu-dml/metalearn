from test.metalearn.metafeatures.test_metafeatures import main as mf_tests, import_openml_dataset, compare_with_openml

def main():
    # mf_tests()
    dataframe, omlMetafeatures = import_openml_dataset()
    compare_with_openml(dataframe, omlMetafeatures)

if __name__ == '__main__':
    main()

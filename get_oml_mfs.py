import openml
import json

file_path = "./oml_metafeatures.json"

oml_datasets = list(openml.datasets.list_datasets().keys())
oml_mfs = set()
for dataset in oml_datasets:
	try:
		print(dataset)
		oml_mfs.update(set(openml.datasets.get_dataset(dataset).qualities.keys()))
		with open(file_path,'w') as fh:
			json.dump(sorted(list(oml_mfs)), fh, indent=4)
	except Exception:
		pass
for mf in oml_mfs:
	print(mf)
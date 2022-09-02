splits="train val test"
for split in $splits; do
	prefix=dataset
	python3 panocutter.py ${prefix}/${split}/panos ${prefix}/${split}/images
	python3 image_to_msgpack.py --output ${prefix}/${split}/msgpack --url_csv ${prefix}/${split}/${split}.csv --image_prefix ${prefix}/${split}/images/
	python3 image_to_country.py \
		--output ${prefix}/${split}/label_mapping \
		--url_csv ${prefix}/${split}/${split}.csv \
		--s2_cells_csv ${prefix}/s2_cells/countries.csv \
		--pseudo_labels ${prefix}/pseudo_labels/countries.json \
		--guidebook ${prefix}/guidebook.json
done
for year in 2017 2018 2019 2020; do 
	echo 'Update data for year' ${year};  
	python scripts/data/extract_nodes.py	--source years/years_${year}.csv \
						--output samples \
						--threshold 12 	
done 

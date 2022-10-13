export MPL_STYLE=seaborn-whitegrid
sources=''
for year in 2017 2018 2019 2020; do 
       sources=${sources}'graphs/wood_years_'${year}'/wood.graphml'
       if [ ${year} -ne 2020 ]; then 
	       sources=${sources}',' 
	fi 
done 
echo 'Sources:' ${sources} 
python scripts/pipeline.py --source ${sources} --output evaluate

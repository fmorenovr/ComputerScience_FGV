for fname in $(dir *.zip); do 
	echo 'Uploading' ${fname} 'to OSF; check https://osf.io/39xks/?view_only=56aa038749fb4e63b98fdc6af7d3db38.'
	osf upload ${fname} ${fname} --force 
done 

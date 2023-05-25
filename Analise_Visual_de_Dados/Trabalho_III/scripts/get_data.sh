mkdir -p data

if ! command -v gdown &> /dev/null; then 
	pip install gdown 
fi

cd data

gdown https://drive.google.com/u/0/uc?id=14BjPYoMpbmIB7CXGu3WazpLHxuX-EBUA&export=download


# Download dataset from Google Drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-81VQNPohLFliCa5ZLGy3TYdMEegBGrn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-81VQNPohLFliCa5ZLGy3TYdMEegBGrn" -O HW2-1-data.tar.gz && rm -rf /tmp/cookies.txt

# Unzip the dataset
tar zxvf HW2-1-data.tar.gz --no-same-owner > log && rm log

# Move the dataset
mv Corpus data

# Download dataset from Google Drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1EKEU3VWr8D5aDKxqx_21lok6oBOQln1X' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1EKEU3VWr8D5aDKxqx_21lok6oBOQln1X" -O HW2-2-data.tar.gz && rm -rf /tmp/cookies.txt

# Unzip the dataset
tar zxvf HW2-2-data.tar.gz --no-same-owner > log && rm log
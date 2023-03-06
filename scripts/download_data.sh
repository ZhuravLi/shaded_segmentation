# downloading dataset
ZIP_FILE=./dataset.zip
FILE_ID=12FO7xaM9yC7Vfr_IDOD-y21ioxlcFlaH
URL=https://drive.google.com/uc?export=download&id=12FO7xaM9yC7Vfr_IDOD-y21ioxlcFlaH

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$FILE_ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILE_ID" -O $ZIP_FILE && rm -rf /tmp/cookies.txt
unzip $ZIP_FILE -d ./dataset
rm $ZIP_FILE

# downloading guideline
ZIP_FILE=./guidelines_of_first_20_images.zip
FILE_ID=11Cc-AXoB_JtH3Y0rCSjZAddQ3qD3DS7p
URL=https://drive.google.com/uc?export=download&id=11Cc-AXoB_JtH3Y0rCSjZAddQ3qD3DS7p

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$FILE_ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILE_ID" -O $ZIP_FILE && rm -rf /tmp/cookies.txt
unzip $ZIP_FILE -d ./guidelines_of_first_20_images
rm $ZIP_FILE
# downloading dataset
ZIP_FILE=./result_imgs.zip
FILE_ID=1cQmntUh09hrgKsgQd2VRuwGRiRncJqwj
URL=   https://drive.google.com/uc?export=download&id=1cQmntUh09hrgKsgQd2VRuwGRiRncJqwj

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$FILE_ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILE_ID" -O $ZIP_FILE && rm -rf /tmp/cookies.txt
unzip $ZIP_FILE -d ./result_imgs
rm $ZIP_FILE

#!/bin/bash
# ---- HMDB51
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar

mkdir -p datasets/HMDB/video_data datasets/HMDB/test_train_splits
unrar e test_train_splits.rar datasets/HMDB/test_train_splits
rm test_train_splits.rar
unrar e hmdb51_org.rar datasets/HMDB/video_data 
rm hmdb51_org.rar


target="datasets/HMDB/video_data"
for f in "$target"/*
do
    FILENAME=(`echo $(basename $f) | tr "." "\n"`)
    echo ${FILENAME}
    rm  $target/$(basename $f)
    mkdir -p $target/${FILENAME}
    unrar e $target/$(basename $f) $target/${FILENAME}
    rm  $target/$(basename $f)  
done

# --- UCF101
wget https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar --no-check-certificate
wget https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate

wget https://www.crcv.ucf.edu/wp-content/uploads/2019/06/Datasets_UCF101-VideoLevel.zip --no-check-certificate


mkdir -p datasets/UCF101/video_data datasets/UCF101/t
est_train_splits
unrar e UCF101.rar datasets/UCF101/video_data
rm UCF101.rar
unzip  UCF101TrainTestSplits-RecognitionTask.zip  datasets/UCF101/ 
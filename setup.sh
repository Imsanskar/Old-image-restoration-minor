cd webpage/
if [ ! -d "./bringing_old_photos_back_to_life" ]
then
    # clone the repo
    git clone https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life.git


    # rename folder
    mv Bringing-Old-Photos-Back-to-Life/ bringing_old_photos_back_to_life/

    #remove cloned folder
    rm -rf Bringing-Old-Photos-Back-to-Life/

    # copy model.py
    cp microsoft_model.py bringing_old_photos_back_to_life/model.py

    cd bringing_old_photos_back_to_life/

    # Clone the Synchronized-BatchNorm-PyTorch repository
    cd Face_Enhancement/models/networks/
    git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
    cd ../../../

    cd Global/detection_models
    git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
    cd ../../

    echo "Downloading the landmark detection pretrained model"
    cd Face_Detection/
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
    cd ../

    # Downloading the pretrained model from Azure Blob Storage
    cd Face_Enhancement/
    wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/face_checkpoints.zip
    unzip face_checkpoints.zip

    # clean the zip file
    rm face_checkpoints.zip
    cd ../
    cd Global/
    wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip
    unzip global_checkpoints.zip

    # clean zip file
    rm global_checkpoints.zip
    cd ../
else
    echo "Old repo found"
fi
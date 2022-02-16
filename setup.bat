@echo off
cd webpage\

if not exist ".\bringing_old_photos_back_to_life" goto SETUP
if exist ".\Bringing-Old-Photos-Back-to-Life" goto RENAME_FOLDER

echo Root directory already exist, skipping cloning
goto SkipRootClone

:SETUP
echo Starting Setup
git clone https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life.git

:RENAME_FOLDER
ren Bringing-Old-Photos-Back-to-Life bringing_old_photos_back_to_life




:SkipRootClone
cd bringing_old_photos_back_to_life\
if exist ".\Face_Enhancement\models\networks\Synchronized-BatchNorm-PyTorch\" goto SkipFaceEnhancementNetwork
echo Downloading Synchronized BatchNorm pytorch in Face_Enhancement
cd Face_Enhancement\models\networks\
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
xcopy /s/e Synchronized-BatchNorm-PyTorch\sync_batchnorm .
cd ..\..\..\

:SkipFaceEnhancementNetwork
if exist ".\Global\detecion_models\Synchronized-BatchNorm-PyTorch\" goto SkipDetectionNetwork
echo Downloading Synchronized BatchNorm pytorch in Face_Detection
cd Global\detection_models
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
xcopy /s/e Synchronized-BatchNorm-PyTorch\sync_batchnorm .
cd ..\..\

:SkipDetectionNetwork
if exist ".\Face_Detection\shape_predictor_68_face_landmarks.dat.bz2" goto SkipShapePredictor
echo Downloading the landmark detection pretrained model
cd Face_Detection/
curl http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 --output shape_predictor_68_face_landmarks.dat.bz2
where 7z >nul 2>nul
IF %ERRORLEVEL% NEQ 0 goto NotFound7z
7z x shape_predictor_68_face_landmarks.dat.bz2
cd ../

:SkipShapePredictor
if exist ".\Face_Enhancement\face_checkpoints" goto SkipFaceCheckPoint
echo Downloading the pretrained model from Azure Blob Storage
cd Face_Enhancement/
curl https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/face_checkpoints.zip --output face_checkpoints.zip
@REM tar -xvf face_checkpoints.zip -C face_checkpoints
@REM del face_checkpoints.zip
cd ../

:SkipFaceCheckPoint
if exist ".\Global\global_checkpoints" goto END
cd Global/
curl https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip --output global_checkpoints.zip
@REM tar -xvf global_checkpoints.zip -C global_checkpoints

@REM del global_checkpoints.zip
goto END

:NotFound7z
echo 7z required to for extracting files, install 7z from here https://www.7-zip.org/
goto END

:END
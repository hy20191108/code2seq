@echo off
REM ###########################################################
REM # Change the following values to train a new model.
REM # type: the name of the new model, only affects the saved file name.
REM # dataset: the name of the dataset, as was preprocessed using preprocess.sh
REM # test_data: by default, points to the validation set, since this is the set that
REM #   will be evaluated after each training iteration. If you wish to test
REM #   on the final (held-out) test set, change 'val' to 'test'.
set type=java-large-model
set dataset_name=java-large
set data_dir=data/java-large
set data=%data_dir%/%dataset_name%
set test_data=%data_dir%/%dataset_name%.val.c2s
set model_dir=models/%type%

if not exist "%model_dir%" mkdir "%model_dir%"
setlocal enabledelayedexpansion
rye run python -u code2seq.py --data %data% --test %test_data% --save_prefix %model_dir%/model
endlocal
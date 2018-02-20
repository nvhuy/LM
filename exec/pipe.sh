#!/bin/bash
echo "[EXMP] Example command > ./pipe.sh LM-FOLDER"
echo "[INFO] For help, enter > ./pipe.sh LM-FOLDER --help"

echo "${@:1}"

if [ "$#" -le 0 ]; then
    echo "[ERRR] Required parameter <path-to-LM-FOLDER> was not found. Exit with code 1"
    exit 1
fi

if [ ! -d "$1" ]; then
    echo "[ERRR] Folder $1 does not exist. Exit with code 2"
    exit 2
fi

PYTHONPATH=$1 python $1/src/spokenCALL/pipeline.py "${@:2}"
#!/usr/bin/env bash

if [[ -z "$@" ]]; then
    echo "Extract following file formats: zip,tar.gz,tgz,rar,tar.bz2."
    # ${0##*/} - filename
    echo "Usage: ${0##*/} [-r] <archive>"
    exit
fi

required_installations=(tar unrar unzip)
for p in ${required_installations[@]}; do
    hash "$p" 2>&- || { echo >&2 " Required program \"$p\" not installed."; exit 1; }
done

# check if file exists
if [ ! -f "${@: -1}" ]; then
    echo "File '${@: -1}' doesn't exist"
    echo "Usage: ${0##*/} [-r] <archive>"
    exit
fi

case "${@: -1}" in
    *.tar.bz2) tar xvjf "${@: -1}" ;;
    *.tar.gz ) tar xvf "${@: -1}" ;;
    *.tar) tar xvf "${@: -1}" ;;
    *.rar ) unrar x "${@: -1}" ;;
    *.zip ) unzip "${@: -1}" ;;
    * ) echo "Unsupported file format" ;;
esac

if [ "$1" = "-r" ]; then
    echo "Removing '${@: -1}'"
    rm ${@: -1}
fi

#!/usr/bin/env bash
file="$1"
NUM_WORDS="$2"

if [[ -z $NUM_WORDS ]]; then
    echo "Usage: '${0##*/}' file.txt NUM_WORDS"
    exit
fi

# check if file exists
if [ ! -f $file ]; then
    echo "File '$file' doesn't exist"
    echo "Usage: '${0##*/}' file.txt NUM_WORDS"
    exit
fi

cat $file | tr '[:upper:]' '[:lower:]' | grep -oE '\w+' | sort | uniq -c | sort -nr | head -n $NUM_WORDS

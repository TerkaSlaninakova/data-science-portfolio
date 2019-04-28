#!/usr/bin/env bash
#
# body: apply expression to all but the first line.
# Use multiple times in case the header spans more than one line.
# 
# Example usage:
# $ seq 10 | header -a 'values' | body sort -nr
# $ seq 10 | header -a 'multi\nline\nheader' | body body body sort -nr
#
# From: http://unix.stackexchange.com/a/11859
#
# See also: header (https://github.com/jeroenjanssens/command-line-tools-for-data-science)
IFS= read -r header
printf '%s\n' "$header"
eval $@
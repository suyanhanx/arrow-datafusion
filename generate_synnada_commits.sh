#!/bin/bash

FILE_NAME="SYNNADA-CONTRIBUTIONS.txt"
git log --pretty=format:"%H" -n $1 --skip 1 > $FILE_NAME

echo "last $1 commits are written to the $FILE_NAME"

#!/bin/bash
filename=$1
aggBagWordsFile=$2
#echo "$filename"
#awkedFile=$filename+"awked"

#awk 'NF && !seen[$0]++' $filename > $awkedFile

#for path in $filename; do
#    name="${path##*/}"
    awk 'NF && !seen[$0]++' $filename   | tr -cd "[:alpha:][:space:]-'" |   tr ' [:upper:]' '\n[:lower:]' |   tr -s '\n' |   sed "s/^['-]*//;s/['-]$//" |   sort |   uniq -c >> $aggBagWordsFile
    awk '{ print $2 }' $aggBagWordsFile > allWords.txt
    sort -u allWords.txt > sortedUniqWords.txt
#remove form feed ^L --> https://unix.stackexchange.com/questions/219438/remove-the-l-aka-f-ff-form-feed-page-break-character
    tr -d '\f' < sortedUniqWords.txt > allUnigrams.txt
    #>> "$dir2/$name"
#done

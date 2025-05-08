#!/bin/bash

if [ -z $WORKDIR ]; then
  printf "\e[31m%s\n\e[m" "Please set WORKDIR environment variable" >&2
  exit 1
fi

delete_items=(
  $WORKDIR/src/model/DEAM/*
  $WORKDIR/src/model/PMEmo/*
)

printf "\e[1;31m%s\e[m \e[1;32m%s\n\e[m" \
      "[WARNING]" "The following directories/files will be deleted"
for delete_item in ${delete_items[@]}; do
    printf "\e[1;34m%s\n\e[m" $delete_item
done

read -n 1 -p "OK?[Y/n] > " confirm
if [[ ${confirm,,} = y ]]; then
  for delete_item in ${delete_items[@]}; do
      if [ -d $delete_item ];then  rm -r $delete_item; fi
      if [ -e $delete_item ];then  rm $delete_item; fi
  done
  printf "\n\e[1;32m%s\n\e[m" "clean done."
else
  printf "\n\e[1;31m%s\n\e[m" "canceled."
fi

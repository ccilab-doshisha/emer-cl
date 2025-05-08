#!/bin/bash -e

USER_ID=$(id -u)
GROUP_ID=$(id -g)

# create group
if [ x"$GROUP_ID" != x"0" ]; then
    groupadd -g $GROUP_ID $USER_NAME
fi

# create user
if [ x"$USER_ID" != x"0" ]; then
    useradd -d /home/$USER_NAME -m -s /bin/bash -u $USER_ID -g $GROUP_ID $USER_NAME
fi

sudo pip install -U pip scikit-learn pandas tqdm matplotlib pysen[lint] nptyping resampy tf_slim six soundfile ffmpeg-python pydub

cp /tmp/bashrc /home/$USER_NAME/.bashrc
sudo chmod u-s /usr/sbin/useradd
sudo chmod u-s /usr/sbin/groupadd
exec $@
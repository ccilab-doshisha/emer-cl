ARG docker_tag
FROM tensorflow/tensorflow:${docker_tag}

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    sudo git vim less tree vim wget ffmpeg && \
    rm -rf /var/lib/apt/list/* /var/cache/apt/archives/* 
RUN apt-get clean

# developer user
ENV USER_NAME=developer
ENV WORKDIR=/work

# for sudo
RUN echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/${USER_NAME}

# for user/group add without sudo
RUN chmod u+s /usr/sbin/useradd \
    && chmod u+s /usr/sbin/groupadd

COPY ./setting/bashrc /tmp/bashrc
COPY ./setting/entrypoint.sh /build/entrypoint.sh
ENTRYPOINT ["/build/entrypoint.sh"]
CMD ["bash"]
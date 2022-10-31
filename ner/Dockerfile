FROM centos/python-38-centos7

USER 0
RUN mkdir -p /apps/ner
WORKDIR /apps/ner

COPY env env
COPY bin/run.sh bin/run.sh

CMD ["bin/run.sh"]

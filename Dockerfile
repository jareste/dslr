FROM python:3.10-slim

WORKDIR /usr/src/app

RUN pip install --no-cache-dir matplotlib numpy

COPY . .

RUN mkdir /output

RUN chmod 777 /output

RUN chmod +x exec_all.sh

CMD ["/bin/sh", "./exec_all.sh"]

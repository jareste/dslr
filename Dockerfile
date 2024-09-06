FROM python:3.10-slim

WORKDIR /usr/src/app

RUN pip install --no-cache-dir matplotlib

COPY . .

RUN mkdir /output

RUN chmod 777 /output

# CMD ["python", "./plot.py"]

CMD ["python", "srcs/describe.py", "datasets/dataset_train.csv"]

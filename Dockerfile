FROM python:3.11-alpine

# ENV PYTHON_VERSION 3.12.2

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN python --version

RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev
RUN apk add libc-dev

RUN python -m pip install --upgrade pip

RUN pip install --upgrade setuptools

RUN pip install -r /app/requirements.txt

COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
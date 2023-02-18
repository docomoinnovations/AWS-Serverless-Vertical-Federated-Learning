FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ./
RUN pip3 install -r requirements.txt
COPY tr_uid.pt ./
COPY tr_y.pt ./
COPY va_uid.pt ./
COPY va_y.pt ./
COPY server_training.py ./

CMD ["server_training.lambda_handler"]
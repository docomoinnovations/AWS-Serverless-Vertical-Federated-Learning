FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ./
RUN pip3 install -r requirements.txt
COPY tr_uid.pt ./
COPY va_uid.pt ./
COPY serverinit.py ./

CMD ["serverinit.lambda_handler"]
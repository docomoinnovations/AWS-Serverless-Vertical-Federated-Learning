FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ./
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt && pip3 uninstall -y pip setuptools
COPY tr_uid.npy ./
COPY tr_y.npy ./
COPY va_uid.npy ./
COPY va_y.npy ./
COPY server_training.py ./

CMD ["server_training.lambda_handler"]
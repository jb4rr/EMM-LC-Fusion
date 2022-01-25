FROM pytorch/pytorch

COPY . /app
RUN pip install -r /app/requirements.txt

CMD python /app/src/main.py
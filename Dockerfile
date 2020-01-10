from python:3.6-slim

RUN apt-get update && apt-get -q -y install --reinstall build-essential && apt-get -q -y install ffmpeg gcc

COPY requirements_1st.txt /app/requirements_1st.txt
COPY requirements_2nd.txt /app/requirements_2nd.txt

RUN pip install --upgrade pip
RUN pip install -r /app/requirements_1st.txt
Run pip install -r /app/requirements_2nd.txt

COPY . /app

WORKDIR /app

ENTRYPOINT ["/bin/bash"]
#ENTRYPOINT ["jupyter", "notebook", "--no-browser", "--allow-root", "--ip=0.0.0.0", "--port=8888"]
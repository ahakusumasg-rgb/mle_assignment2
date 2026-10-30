FROM apache/airflow:3.3.2

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre-headless procps libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s "$(dirname "$(dirname "$(readlink -f "$(which java)")")")" /opt/java
ENV JAVA_HOME=/opt/java

USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
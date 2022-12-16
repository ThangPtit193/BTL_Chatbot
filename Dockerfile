FROM silverlogic/python3.6
ENV APPLICATION_SERVICE = /saturn
ENV PYTHONPATH "${PYTHONPATH}:$APPLICATION_SERVICE"

EXPOSE 8501:8501

WORKDIR $APPLICATION_SERVICE

RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools
RUN pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install requirements
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY . ./
COPY ./.env ./.env
RUN axiom login --email 'phongnt@ftech.ai' --password 'b8dJfQFq6DL3'
ENTRYPOINT ["streamlit", "run", "ui/navigation.py"]
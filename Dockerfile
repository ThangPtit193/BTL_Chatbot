FROM silverlogic/python3.6
ENV PYTHONPATH "${PYTHONPATH}/saturn"
ENV APPLICATION_SERVICE = /saturn
ARG AIXOM_EMAIL
ARG AXIOM_PASSWORD

EXPOSE 8501:8501

WORKDIR $APPLICATION_SERVICE

RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools
RUN pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install requirements
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY . ./
COPY ./template.env ./.env
RUN axiom login --email $AXIOM_EMAIL --password $AXIOM_PASSWORD
ENTRYPOINT ["streamlit", "run", "ui/navigation.py"]
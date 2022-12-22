FROM silverlogic/python3.7
ENV APPLICATION_SERVICE = /saturn
ENV PYTHONPATH "${PYTHONPATH}:$APPLICATION_SERVICE"
ARG AXIOM_EMAIL
ARG AXIOM_PASSWORD
ENV email=$AXIOM_EMAIL
ENV password=$AXIOM_PASSWORD

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
RUN echo 'Log in Axiom with user' $AXIOM_EMAIL
RUN axiom login --email $AXIOM_EMAIL --password $AXIOM_PASSWORD
RUN echo 'Welcome to Knowledge Retrieval Services'
ENTRYPOINT ["streamlit", "run", "ui/navigation.py"]
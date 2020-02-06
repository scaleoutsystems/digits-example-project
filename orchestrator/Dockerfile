FROM python:3.7

RUN mkdir /app
COPY . /app/
WORKDIR /app

RUN mkdir /dep
RUN chmod 600 fetch_key
RUN ssh-agent bash -c "ssh-add fetch_key;GIT_SSH_COMMAND='ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no' git clone git@github.com:scaleoutsystems/scaleout-cli.git /dep/scaleout-cli"
RUN pip install -e /dep/scaleout-cli
RUN pip install -r requirements.txt

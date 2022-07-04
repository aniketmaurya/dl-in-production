FROM python:3.9

# copy fastapi app
COPY requirements.txt requirements.txt
COPY app.py app.py

# Install dependencies
RUN pip install -U torch "fastapi[standard]"

# Expose port 80 to accesible from outside the container
EXPOSE 80
ENTRYPOINT "uvicorn app:app --ip 0.0.0.0 --port 80"

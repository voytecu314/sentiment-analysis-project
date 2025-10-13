# Layer 1: Use an official Python image as a base image
FROM python:3.11-slim

# Layer 2: Set the working directory inside the container
WORKDIR /app

# Layers 3&4: Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layers 5&6: Copy local project files into the container
# The first path is the source on your local machine.
# The second path is the destination inside the container's /app directory.
COPY src/ src/
COPY data/ data/

RUN python src/train.py --data data/sentiments.csv --out models/sentiment.joblib
# Layer 7: Specify the command to run on container start
CMD ["python", "src/predict.py", "That was the best", "I'm happy", "Whoa! That's terrible", "That's so bad", "That was really anoying"]

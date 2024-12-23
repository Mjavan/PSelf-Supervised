# Step 1: Use an official Python runtime as the base image
FROM python:3.12-slim

ENV PYTHONBUFFERED=TRUE

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY requirements.txt /app

# Step 4: Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt


COPY core/ /app/core
COPY run.sh /app/core
RUN chmod +x /app/core/run.sh

ENTRYPOINT ["/app/core/run.sh"]

# Step 5: Make port 8000 available to the outside world
EXPOSE 8000

CMD ["--help"]  # Default argument for entrypoint.sh
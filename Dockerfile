FROM python:3.10-slim
# Set the working directory in the container
WORKDIR /app

# Copy the Python script into the container
COPY StreamDiffusion/examples/screen/final_server.py /app/
COPY StreamDiffusion/examples/screen/requirements.txt /app/
# Install any dependencies required by your script
RUN pip install -r requirements.txt

# Specify the command to run your script
CMD [ "python", "./final_server.py" ]
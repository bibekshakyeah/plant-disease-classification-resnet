FROM pytorch/pytorch:latest

# Install Jupyter and torchsummary
RUN pip install jupyter torchsummary numpy pandas matplotlib Pillow torchvision

# Set the working directory to /app
WORKDIR /app

# Copy the contents of the local directory to /app in the container
COPY . /app

# Expose port 8888 for Jupyter notebooks
EXPOSE 8888

# Start Jupyter notebook on container start
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]

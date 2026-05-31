FROM python:3.11-slim

# System settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Create a non-root user (Hugging Face Spaces runs as UID 1000)
RUN useradd -m -u 1000 user

# Set up working directory inside the non-root user home
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install dependencies as root, but caching the layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and set ownership to the non-root user
COPY --chown=user . $HOME/app

# Switch to the non-root user
USER user

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Command to run uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

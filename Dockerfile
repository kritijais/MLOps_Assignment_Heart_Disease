# Use the official Python base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# --- Install Dependencies ---

# 1. Copy only the requirements file first to leverage Docker caching.
# Assuming you have a requirements.txt generated in your project root.
# If not, create one with: pip freeze > requirements.txt
COPY requirements.txt .

# 2. Install dependencies
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Install packages required for joblib/sklearn (often implicit in full images)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Copy Application and Artifacts ---

# 3. Copy the entire source code (src/ folder)
COPY src/ ./src/

# 4. Copy the trained artifacts (model and preprocessor)
# These MUST be in the project root for model_handler.py to find them.
COPY model.pkl .
COPY preprocessor.pkl .

# --- Expose Port and Define Command ---

# 5. Expose the port where the FastAPI application will run
EXPOSE 8000

# 6. Define the command to run the application using uvicorn
# We specify the application path: src.api.main:app
# Bind to 0.0.0.0 for container networking
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
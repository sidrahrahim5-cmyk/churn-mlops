# Docker File

# Step 1: Base Image
FROM python:3.11-slim

# Step 2: Set working directory
WORKDIR /app

# STep 3: Copy Requirements
COPY requirements.txt .

# Step 4: Install libraries
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy Project Files
COPY . .

# Step 6: Expose ports
EXPOSE 8000

# Step 7: Sart API
CMD ["uvicorn", "src.serving.app:app","--host", "0.0.0.0", "--port", "8000"]

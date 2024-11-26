# Gunakan Python sebagai base image
FROM python:3.9-slim

# Salin semua file ke container
WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan aplikasi Flask
CMD ["python", "main.py"]

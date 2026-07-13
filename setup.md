I'll give you the deployment commands for both local Docker and server deployment.

Quick Deployment Commands
1. Local Docker (Development/Testing)

cd "/home/hout/Documents/Paragon/Semester VI/CS 382 - Search Engines and Information Retrieval/final"

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# GOOGLE_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here

# Build and run
docker-compose up -d --build

# View logs
docker-compose logs -f rag-search

# Stop
docker-compose down
2. Server Deployment (Production)

# On your server, clone or copy the project
cd /opt/rag-search  # or wherever you want it

# Set up environment
cp .env.example .env
# Edit .env with API keys

# Build the image (if not pushing pre-built)
docker build -t rag-search:latest .

# Run with port mapping (example: port 8080)
docker run -d \
  --name rag-search-app \
  --restart unless-stopped \
  -p 8080:8501 \
  --env-file .env \
  rag-search:latest

# Check status
docker ps
docker logs rag-search-app
App runs at http://your-server-ip:8080

3. With Reverse Proxy (Nginx) - Recommended for Production

# Create nginx config
sudo nano /etc/nginx/sites-available/rag-search
Add:


server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
Enable and restart:


sudo ln -s /etc/nginx/sites-available/rag-search /etc/nginx/sites-enabled/
sudo systemctl restart nginx
Access at http://yourdomain.com

4. Push Pre-Built Image to Registry (if using Docker Hub/Private Registry)

# Tag and push
docker tag rag-search:latest yourusername/rag-search:latest
docker push yourusername/rag-search:latest

# On server, pull and run
docker pull yourusername/rag-search:latest
docker run -d --name rag-search-app -p 8080:8501 \
  --env-file .env \
  yourusername/rag-search:latest
5. Check Everything's Working

# Verify container is running
docker ps | grep rag-search

# Check logs for errors
docker logs rag-search-app

# Test the API (should return 200)
curl http://localhost:8501
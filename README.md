# Local LLM Performance Simulator

Web simulator to estimate LLM inference performance on different hardware.

**Live**: [llmsimulation.ht-x.com](https://llmsimulation.ht-x.com)

## Features

- **"I have hardware"**: Select your hardware and see which models run and at what speed
- **"I want a model"**: Select the model and desired speed, see which hardware you need
- ML-based predictions trained on 207+ real benchmarks
- Token streaming animation to visualize generation speed

## Local Development

```bash
# Backend
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

---

## Production Deployment (Linux + NGINX)

### 1. Server Requirements

- Linux (Ubuntu 24.04 LTS recommended)
- Python 3.12+
- Node.js 20+
- NGINX
- Certbot (for SSL)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Python 3.12 (included in Ubuntu 24.04)
sudo apt install python3.12-venv -y

# Node.js 20 (via NodeSource)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install nodejs -y

# NGINX
sudo apt install nginx -y
sudo systemctl enable nginx
sudo systemctl start nginx

# Certbot
sudo apt install certbot python3-certbot-nginx -y

# Verify installations
python3 --version
node --version
nginx -v
certbot --version
```

### 2. Clone and Setup

```bash
# Clone repository
cd /var/www
git clone https://github.com/htxsrl/localllmsimulation.git
cd localllmsimulation

# Backend setup
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend build
cd ../frontend
npm ci
npm run build
```

### 3. Systemd Service for Backend

Create `/etc/systemd/system/llm-simulator.service`:

```ini
[Unit]
Description=LLM Performance Simulator API
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/var/www/localllmsimulation/backend
Environment="PATH=/var/www/localllmsimulation/backend/.venv/bin"
ExecStart=/var/www/localllmsimulation/backend/.venv/bin/uvicorn src.main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable llm-simulator
sudo systemctl start llm-simulator
sudo systemctl status llm-simulator
```

### 4. NGINX Configuration

Create `/etc/nginx/sites-available/llmsimulation.ht-x.com`:

```nginx
server {
    listen 80;
    server_name llmsimulation.ht-x.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name llmsimulation.ht-x.com;

    # SSL (managed by Certbot)
    ssl_certificate /etc/letsencrypt/live/llmsimulation.ht-x.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/llmsimulation.ht-x.com/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Frontend static files
    root /var/www/localllmsimulation/frontend/dist;
    index index.html;

    # Static files caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # API proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Cache API responses (optional)
        proxy_cache_valid 200 5m;
    }

    # Health check
    location /health {
        proxy_pass http://127.0.0.1:8000;
    }

    # SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Gzip
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
    gzip_min_length 1000;
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/llmsimulation.ht-x.com /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 5. SSL Certificate

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate (before enabling NGINX SSL config)
sudo certbot --nginx -d llmsimulation.ht-x.com

# Auto-renewal is enabled by default
sudo certbot renew --dry-run
```

### 6. DNS Configuration

Add an A record in your DNS provider:

```
Type: A
Name: llmsimulation
Value: <your-server-ip>
TTL: 3600
```

### 7. Useful Commands

```bash
# Check backend status
sudo systemctl status llm-simulator
sudo journalctl -u llm-simulator -f

# Restart after updates
cd /var/www/localllmsimulation
git pull
cd frontend && npm ci && npm run build
sudo systemctl restart llm-simulator

# Check NGINX logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

---

## Tech Stack

- **Backend**: FastAPI + SQLite (in-memory) + scikit-learn
- **Frontend**: React + Vite
- **Data**: 207 real benchmarks from llama.cpp, LocalScore.ai, hardware-corner.net

## Author

Francesco Menegoni | [Human Technology eXcellence - HTX SRL](https://ht-x.com)

## License

MIT

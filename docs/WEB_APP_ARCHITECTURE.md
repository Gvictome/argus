# ARGUS Web App Architecture
## Cloud Deployment on Render or Cloudflare

---

## Overview

The ARGUS web app provides users with a dashboard to monitor their home security, view live feeds, receive alerts, and configure automation rules. The app connects to the local Raspberry Pi device via secure tunneling.

---

## Architecture Options

### Option A: Render Deployment (Recommended for MVP)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RENDER CLOUD                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│  │   Web Service   │    │   API Service   │    │   PostgreSQL   │  │
│  │   (Frontend)    │◄──►│   (Backend)     │◄──►│   Database     │  │
│  │   React/Next.js │    │   FastAPI       │    │                │  │
│  │   Static Site   │    │   Python        │    │   Users        │  │
│  └─────────────────┘    └─────────────────┘    │   Devices      │  │
│          │                      │              │   Events       │  │
│          │                      │              └────────────────┘  │
│          │                      │                                  │
│          │              ┌───────┴───────┐                         │
│          │              │    Redis      │                         │
│          │              │   (Caching)   │                         │
│          │              └───────────────┘                         │
│          │                      │                                  │
└──────────┼──────────────────────┼──────────────────────────────────┘
           │                      │
           │         ┌────────────┴────────────┐
           │         │   Cloudflare Tunnel     │
           │         │   (Secure Connection)   │
           │         └────────────┬────────────┘
           │                      │
           ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      USER'S HOME NETWORK                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    RASPBERRY PI 5                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │   Camera    │  │   AI HAT+   │  │   Local API         │  │   │
│  │  │   Stream    │  │   Detection │  │   (FastAPI)         │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Render Services:**
| Service | Type | Cost |
|---------|------|------|
| Frontend | Static Site | Free |
| Backend API | Web Service | $7/mo |
| PostgreSQL | Database | Free (90 days) / $7/mo |
| Redis | Key-Value Store | $0 (use Upstash free tier) |

---

### Option B: Cloudflare Deployment (Better Performance)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CLOUDFLARE EDGE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│  │  Cloudflare     │    │   Cloudflare    │    │   Cloudflare   │  │
│  │  Pages          │◄──►│   Workers       │◄──►│   D1 Database  │  │
│  │  (Frontend)     │    │   (API)         │    │   (SQLite)     │  │
│  │  React/Next.js  │    │   Edge Runtime  │    │                │  │
│  └─────────────────┘    └─────────────────┘    └────────────────┘  │
│          │                      │                      │           │
│          │                      │              ┌───────┴────────┐  │
│          │                      │              │   KV Storage   │  │
│          │                      │              │   (Sessions)   │  │
│          │                      │              └────────────────┘  │
│          │                      │                                  │
│          │              ┌───────┴───────┐                         │
│          │              │  Durable      │                         │
│          │              │  Objects      │                         │
│          │              │  (WebSocket)  │                         │
│          │              └───────────────┘                         │
│          │                      │                                  │
│          │         ┌────────────┴────────────┐                    │
│          │         │   Cloudflare Tunnel     │                    │
│          │         │   (Zero Trust)          │                    │
│          │         └────────────┬────────────┘                    │
└──────────┼──────────────────────┼──────────────────────────────────┘
           │                      │
           ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RASPBERRY PI 5 (User's Home)                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │   cloudflared daemon → Local ARGUS API (localhost:8000)     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**Cloudflare Services:**
| Service | Type | Cost |
|---------|------|------|
| Pages | Static Site | Free |
| Workers | Serverless API | Free (100k req/day) |
| D1 | SQLite Database | Free (5GB) |
| KV | Key-Value Store | Free (100k reads/day) |
| Tunnel | Secure Connection | Free |
| Durable Objects | WebSocket/State | $0.15/million requests |

---

## Recommendation

| Factor | Render | Cloudflare |
|--------|--------|------------|
| **Setup Complexity** | Easy | Medium |
| **Performance** | Good | Excellent (Edge) |
| **Cost (Start)** | ~$7/mo | Free |
| **Cost (Scale)** | $15-50/mo | $5-20/mo |
| **WebSocket** | Native | Durable Objects |
| **Live Streaming** | Works | Better (Edge caching) |
| **Learning Curve** | Lower | Higher |

**Recommendation:** Start with **Cloudflare** for cost efficiency and edge performance, especially important for live video streaming.

---

## Tech Stack

### Frontend (Both Platforms)

```
┌─────────────────────────────────────────────────────────────────┐
│                      FRONTEND STACK                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Framework:     Next.js 14 (App Router)                       │
│   Language:      TypeScript                                     │
│   Styling:       TailwindCSS + shadcn/ui                       │
│   State:         Zustand or Jotai                              │
│   Data Fetching: TanStack Query (React Query)                  │
│   Video Player:  HLS.js for live streaming                     │
│   WebSocket:     Native WebSocket / Socket.io                  │
│   Auth:          NextAuth.js / Clerk                           │
│   PWA:           next-pwa for mobile install                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Backend API

**Render (Python):**
```
FastAPI + SQLAlchemy + PostgreSQL + Redis
```

**Cloudflare (Edge):**
```
Hono.js (Workers) + Drizzle ORM + D1 + KV
```

---

## Database Schema

```sql
-- Users table
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Devices (Raspberry Pi units)
CREATE TABLE devices (
    id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(id),
    name TEXT NOT NULL,
    tunnel_id TEXT UNIQUE,          -- Cloudflare tunnel ID
    status TEXT DEFAULT 'offline',  -- online/offline
    last_seen TIMESTAMP,
    config JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Events (detections, alerts)
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    device_id TEXT REFERENCES devices(id),
    type TEXT NOT NULL,             -- person, face, motion, zone
    confidence REAL,
    thumbnail_url TEXT,
    video_clip_url TEXT,
    metadata JSON,                  -- bounding boxes, face ID, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Known Faces
CREATE TABLE faces (
    id TEXT PRIMARY KEY,
    device_id TEXT REFERENCES devices(id),
    name TEXT NOT NULL,
    embedding BLOB,                 -- Face embedding vector
    photo_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Zones
CREATE TABLE zones (
    id TEXT PRIMARY KEY,
    device_id TEXT REFERENCES devices(id),
    name TEXT NOT NULL,
    type TEXT DEFAULT 'alert',      -- alert, ignore, restricted
    polygon JSON,                   -- [[x,y], [x,y], ...]
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Automation Rules
CREATE TABLE rules (
    id TEXT PRIMARY KEY,
    device_id TEXT REFERENCES devices(id),
    name TEXT NOT NULL,
    trigger JSON,                   -- {type: 'person', zone: 'front_door'}
    action JSON,                    -- {type: 'notify', channels: ['telegram']}
    schedule JSON,                  -- {days: [1,2,3,4,5], start: '09:00', end: '17:00'}
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## App Pages & Features

### User Dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│  ARGUS                                    [Settings] [Profile]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────┐  ┌─────────────────────────┐  │
│  │                             │  │  LIVE STATUS            │  │
│  │      LIVE CAMERA FEED       │  │  ────────────────────   │  │
│  │         (HLS Stream)        │  │  ● Device: Online       │  │
│  │                             │  │  ● Detection: Active    │  │
│  │    [Person Detected]        │  │  ● Recording: Ready     │  │
│  │    Confidence: 97%          │  │  ● Last Event: 2m ago   │  │
│  │                             │  │                         │  │
│  └─────────────────────────────┘  └─────────────────────────┘  │
│                                                                 │
│  RECENT EVENTS                                                  │
│  ─────────────────────────────────────────────────────────────  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ 2:34 PM │ │ 2:15 PM │ │ 1:45 PM │ │ 1:30 PM │ │ 1:12 PM │  │
│  │ Person  │ │ Unknown │ │ Package │ │ Motion  │ │ John    │  │
│  │ [thumb] │ │ [thumb] │ │ [thumb] │ │ [thumb] │ │ [thumb] │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
│                                                                 │
│  [View All Events]                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### App Routes

| Route | Description |
|-------|-------------|
| `/` | Landing page |
| `/login` | Authentication |
| `/register` | New user signup |
| `/dashboard` | Main dashboard with live feed |
| `/events` | Event history with filters |
| `/events/[id]` | Event detail with video clip |
| `/zones` | Zone configuration |
| `/faces` | Face enrollment & management |
| `/rules` | Automation rules |
| `/settings` | Device & notification settings |
| `/settings/device` | Device configuration |

---

## API Endpoints

### Authentication
```
POST   /api/auth/register     # New user
POST   /api/auth/login        # Login
POST   /api/auth/logout       # Logout
GET    /api/auth/me           # Current user
```

### Devices
```
GET    /api/devices           # List user's devices
POST   /api/devices           # Register new device
GET    /api/devices/:id       # Device details
PATCH  /api/devices/:id       # Update device
DELETE /api/devices/:id       # Remove device
GET    /api/devices/:id/stream  # Live stream URL
```

### Events
```
GET    /api/events            # List events (paginated)
GET    /api/events/:id        # Event detail
DELETE /api/events/:id        # Delete event
GET    /api/events/:id/clip   # Video clip URL
```

### Zones
```
GET    /api/zones             # List zones
POST   /api/zones             # Create zone
PATCH  /api/zones/:id         # Update zone
DELETE /api/zones/:id         # Delete zone
```

### Faces
```
GET    /api/faces             # List known faces
POST   /api/faces             # Enroll new face
PATCH  /api/faces/:id         # Update face name
DELETE /api/faces/:id         # Remove face
```

### Rules
```
GET    /api/rules             # List automation rules
POST   /api/rules             # Create rule
PATCH  /api/rules/:id         # Update rule
DELETE /api/rules/:id         # Delete rule
```

### WebSocket
```
WS     /api/ws                # Real-time events
```

---

## Device-to-Cloud Communication

### Cloudflare Tunnel Setup (on Raspberry Pi)

```bash
# Install cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 -o cloudflared
chmod +x cloudflared
sudo mv cloudflared /usr/local/bin/

# Authenticate
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create argus-home

# Configure tunnel (config.yml)
tunnel: <TUNNEL_ID>
credentials-file: /home/pi/.cloudflared/<TUNNEL_ID>.json
ingress:
  - hostname: device-<USER_ID>.argus.app
    service: http://localhost:8000
  - hostname: stream-<USER_ID>.argus.app
    service: http://localhost:8080
  - service: http_status:404

# Run as service
sudo cloudflared service install
sudo systemctl start cloudflared
```

### Event Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Pi Camera   │────▶│  Detection   │────▶│  Event       │
│  (30 FPS)    │     │  (YOLOv8)    │     │  Generated   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌────────────────────────────┤
                     │                            │
                     ▼                            ▼
              ┌──────────────┐           ┌──────────────┐
              │  Local Save  │           │  Cloud Push  │
              │  (SQLite)    │           │  (WebSocket) │
              └──────────────┘           └──────┬───────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │  Cloudflare  │
                                         │  Workers API │
                                         └──────┬───────┘
                                                │
                        ┌───────────────────────┼───────────────────────┐
                        │                       │                       │
                        ▼                       ▼                       ▼
                 ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
                 │  D1 Database │       │  Push Notif  │       │  WebSocket   │
                 │  (Store)     │       │  (Telegram)  │       │  (Dashboard) │
                 └──────────────┘       └──────────────┘       └──────────────┘
```

---

## Deployment Commands

### Cloudflare Pages + Workers

```bash
# Install Wrangler CLI
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Deploy frontend (Next.js to Pages)
cd web
npm run build
wrangler pages deploy .next

# Deploy API (Workers)
cd api
wrangler deploy

# Create D1 database
wrangler d1 create argus-db
wrangler d1 execute argus-db --file=schema.sql
```

### Render

```bash
# Use render.yaml for infrastructure as code
# See render.yaml in repo root

# Or deploy via Render Dashboard:
# 1. Connect GitHub repo
# 2. Create Web Service (API)
# 3. Create Static Site (Frontend)
# 4. Create PostgreSQL database
```

---

## Cost Comparison (Monthly)

| Users | Render | Cloudflare |
|-------|--------|------------|
| 1-10 | $7 | $0 |
| 10-100 | $14 | $0-5 |
| 100-1000 | $25-50 | $5-15 |
| 1000+ | $100+ | $20-50 |

---

## Security Considerations

1. **Authentication:** JWT tokens with refresh rotation
2. **Device Auth:** Unique API keys per device
3. **Encryption:** TLS everywhere, encrypted tunnels
4. **Video Privacy:** Stream only to authenticated users
5. **Data Retention:** Configurable auto-delete for events
6. **Rate Limiting:** Protect API from abuse

---

*Last Updated: January 2026*
*Project: ARGUS Senior Design*

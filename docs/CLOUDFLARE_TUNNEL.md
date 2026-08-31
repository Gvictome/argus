# Cloudflare Tunnel

Reaching the ARGUS dashboard and live feed from outside the LAN, without
port forwarding.

> ## Read this before you start
>
> A tunnel publishes a **live camera feed** and the controls to **enroll
> and delete faces**. Exposing that without authentication is not a
> configuration mistake, it is a privacy incident — the camera is pointed
> at real people who did not consent to being on the public internet.
>
> Two gates, both required:
>
> 1. **Cloudflare Access** — authentication at the edge. Unauthenticated
>    requests never reach the Pi.
> 2. **`AUTH_REQUIRED=true`** — ARGUS's own bearer-token auth, so a
>    mistyped Access policy is not the only thing in the way.
>
> `scripts/run_tunnel.sh` refuses to start without gate 2, and proves it
> by checking that a protected endpoint actually returns 401.

---

## 1. Turn on authentication

Auth defaults to **off**, which is fine on a trusted LAN and keeps the
existing demo flow unchanged. The tunnel needs it on.

```bash
export AUTH_REQUIRED=true
export SECRET_KEY="$(openssl rand -hex 32)"
export ARGUS_ADMIN_PASSWORD="a real password, 12+ characters"
python main.py
```

On first run ARGUS creates an `admin` account. If you omit
`ARGUS_ADMIN_PASSWORD` it generates one and prints it **once** as a
warning — record it then, because it is stored only as a PBKDF2 hash.

Forgot it, or inherited a generated one?

```bash
python scripts/set_password.py --user admin
```

Verify:

```bash
curl -s localhost:8000/api/auth/me          # {"auth_required":true,...}
curl -o /dev/null -w '%{http_code}\n' localhost:8000/api/faces   # 401
```

`/api/auth/me` never returns 401 by design — it is how a caller
discovers whether auth is on at all.

---

## 2. Install and create the tunnel

On the Pi:

```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb

cloudflared tunnel login
cloudflared tunnel create argus-demo
cloudflared tunnel route dns argus-demo argus.yourdomain.com
```

---

## 3. Configure

```bash
cp config/cloudflared-argus.yml ~/.cloudflared/config.yml
nano ~/.cloudflared/config.yml     # set hostname + credentials path
```

The MJPEG-specific settings in that file are load-bearing:

| Setting | Why |
|---------|-----|
| `disableChunkedEncoding: true` | Without it cloudflared buffers the MJPEG stream and video stutters or never starts |
| `keepAliveTimeout: 90s` | The stream is one long-lived response, not a series of requests |
| `retries: 5`, `grace-period: 30s` | Survives a wifi blip instead of ending the demo |

---

## 4. Add a Cloudflare Access policy

**Do not skip this.** Without it the hostname is public and only ARGUS's
own auth stands between the internet and the camera.

Zero Trust dashboard → **Access → Applications → Add an application →
Self-hosted**:

- Application domain: `argus.yourdomain.com`
- Policy: *Allow*, with **Emails** matching your team's addresses
- Session duration: short — an hour is plenty for a demo

One-time PIN to a known email is enough and needs no identity provider.

---

## 5. Start it

```bash
scripts/run_tunnel.sh
```

Preflight runs first and refuses on any of:

- ARGUS is not responding
- `AUTH_REQUIRED` is off
- a protected endpoint does **not** return 401
- the config still has the placeholder `yourdomain.com`
- `cloudflared` or the credentials file is missing

Check without starting:

```bash
scripts/run_tunnel.sh --check
```

---

## 6. Run it as a service

```bash
sudo cloudflared service install
sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared
```

Note this starts the tunnel at boot **regardless of whether ARGUS has
auth on** — the systemd unit does not run the preflight. If you install
the service, put `AUTH_REQUIRED=true` in ARGUS's own unit file so the two
cannot drift apart.

---

## 7. Using it

Open `https://argus.yourdomain.com`, clear the Access prompt, then sign
in to ARGUS itself. The dashboard is served at `/` for browsers and at
`/dashboard` explicitly.

---

## 8. Troubleshooting

| Symptom | Cause |
|---------|-------|
| Preflight: "AUTH_REQUIRED is OFF" | Restart ARGUS with `AUTH_REQUIRED=true`. The check asks the running server, not your shell. |
| Preflight: "returned 200, expected 401" | Auth is on but not enforced — check that routes carry `dependencies=_PROTECTED`. |
| 502 from Cloudflare | ARGUS is not listening on 8000. |
| Dashboard loads, video is blank | The stream token. Sign out and back in; check the browser console. |
| Video stutters badly | Confirm `disableChunkedEncoding: true`, then raise `detect_every` on the stream URL. |
| "Invalid username or password" and you are sure | The admin was created earlier with a different password. `scripts/set_password.py`. |
| Tunnel connects, hostname 404s | `ingress` hostname does not match the DNS route. |

---

## 9. What is still missing

Worth saying plainly, because a judge may ask and because
`docs/SHOWCASE_SPRINT_PLAN.md` A-9 calls for exactly this honesty:

- **Tokens live in memory.** A restart logs everyone out. Fine for a
  demo, wrong for production.
- **No rate limiting** on `/api/auth/login`. Cloudflare Access is what
  actually stands between the internet and a brute-force attempt, which
  is another reason gate 1 is not optional.
- **AES-256 at rest is not implemented.** `SecurityService.encrypt()` is
  still a passthrough stub. Face embeddings are stored as unencrypted
  pickles.
- **No token refresh.** Sessions expire at `TOKEN_EXPIRY` (1h default)
  and you sign in again.

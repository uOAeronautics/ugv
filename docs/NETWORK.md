# Network & SSH

Single source of truth for Pi network access and dashboard endpoint.

- **Pi SSH**: `ssh pi` (`advanced@192.168.8.100` via the `glinet` jump host, configured in `~/.ssh/config`).
- **Pi checkout**: `/home/advanced/ugv`; **service**: `ugv-dashboard.service` on port `8765`.
- **Dashboard URL**: `http://192.168.8.100:8765` (port `8765`; forward or access via the `glinet` LAN).
- **Stale addresses**: prior `192.168.8.169`, `192.168.8.170`, and `ssh advanced` aliases are retired.

## NM profile hardening

- `FillMeUpLink` -> DHCP on `wlan0`.
- `autoconnect=yes`; priority `100`; retries `0` (forever).
- Wi-Fi power save `disabled` -> fewer idle/link-drop failures.
- Route metric: Wi-Fi `600`, Ethernet `100` -> Ethernet preferred when cabled, Wi-Fi remains fallback.
- Observed boot association + DHCP ≈ 8 s.

## Reconnect & credential invariants

- Dashboard safety is independent of network reconnect: browser sends fresh drive intent every `150 ms`; hardware guard brakes after `600 ms` without a fresh drive message. Generic WebSocket heartbeat never renews motion.
- No password or private key is stored in this repository.

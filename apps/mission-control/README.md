# Mission Control

Electron desktop app for operating the local Model Express orchestrator.

## Development

Start the backing services and orchestrator first:

```powershell
docker compose -f ..\..\infra\compose.yaml up -d postgres minio
cd ..\..\services\orchestrator
go run ./cmd/orchestrator
```

Then run the desktop app:

```powershell
cd apps/mission-control
npm install
npm run dev
```

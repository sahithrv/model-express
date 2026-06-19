# Mission Control

Electron desktop app for operating the local Model Express orchestrator.

## Development

Mission Control starts the local Postgres and MinIO support services with Docker Compose.
Start Docker Desktop, then start the orchestrator and app:

```powershell
cd ..\..\services\orchestrator
go run ./cmd/orchestrator
cd ..\..\apps\mission-control
npm install
npm run dev
```

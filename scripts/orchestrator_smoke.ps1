$BaseUrl = if ($env:ORCHESTRATOR_URL) { $env:ORCHESTRATOR_URL } else { "http://localhost:8080" }

Write-Host "Checking orchestrator health..."
Invoke-RestMethod "$BaseUrl/healthz"

Write-Host "Creating project..."
$project = Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/projects" `
  -ContentType "application/json" `
  -Body '{"name":"smoke-demo","goal":"combat vs menu classifier"}'

Write-Host "Creating queued job..."
$jobBody = @{
  template = "mobilenet_transfer"
  config = @{
    image_size = 224
    epochs = 3
    learning_rate = 0.0003
  }
} | ConvertTo-Json -Depth 10

$job = Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/projects/$($project.id)/jobs" `
  -ContentType "application/json" `
  -Body $jobBody

Write-Host "Registering worker..."
$workerBody = @{
  project_id = $project.id
  name = "local-smoke-worker-1"
  gpu_type = "local"
} | ConvertTo-Json -Depth 10

$worker = Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/workers/register" `
  -ContentType "application/json" `
  -Body $workerBody

Write-Host "Polling job..."
Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/workers/$($worker.id)/poll"

Write-Host "Reporting fake epoch metric..."
Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/jobs/$($job.id)/metrics" `
  -ContentType "application/json" `
  -Body '{"epoch":1,"metrics":{"train_loss":0.91,"val_loss":0.82,"macro_f1":0.41}}'

Write-Host "Reading job metrics..."
Invoke-RestMethod "$BaseUrl/jobs/$($job.id)/metrics"

Write-Host "Reading project jobs..."
Invoke-RestMethod "$BaseUrl/projects/$($project.id)/jobs"

Write-Host "Reading workers..."
Invoke-RestMethod "$BaseUrl/workers"

Write-Host "Reading project workers..."
Invoke-RestMethod "$BaseUrl/projects/$($project.id)/workers"

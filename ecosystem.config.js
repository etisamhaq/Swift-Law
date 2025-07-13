module.exports = {
  apps: [
    {
      name: "law-gpt",
      script: "streamlit",
      args: "run app.py --server.port 8501 --server.address 0.0.0.0",
      env: {
        NODE_ENV: "production",
        // Add other environment variables here
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "./logs/error.log",
      out_file: "./logs/out.log",
      merge_logs: true,
      time: true
    }
  ]
};
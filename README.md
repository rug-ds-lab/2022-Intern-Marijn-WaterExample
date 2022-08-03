# Leak Detection ECiDA-Vitens

## Instructions
Start the application with the following command:
```shell
docker-compose up -d
```
Data should start flowing into the database, this process can be monitored using Grafana. Grafana can be accessed at:
```shell
http://localhost:3000
```

InfluxDB can also be accessed directly at:
```shell
http://localhost:8086
```

### Additional notes

Currently, the `pipeline1` container cannot run after simply cloning from here since it relies on a model file 
which cannot be committed here due to its size
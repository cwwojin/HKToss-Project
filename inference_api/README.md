# ML model inference API

---

## 1. System Requirements Specification

### 1.1. Functional Requirements

1. Model inference - 1 or more rows, single or ALL model types
2. Model inference - single row, permutation test
3. Auth - API Key (fixed)
4. Health Check - `/health`

### 1.2. Non-functional Requirements

1. Dependencies : `FastAPI, MLFlow==2.15.1`
2. Deployment : `ECS + ALB`
3. Database / Storage : `MongoDB, S3`
4. CI/CD : `Github Actions`

---

## 2. Running the API Server

-   run this in **Project Root**

```shell
$ docker compose --profile api -f docker-compose.yml up
```

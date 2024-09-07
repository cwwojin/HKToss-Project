# Airflow Pipeline

---

## 1. Installation

-   Run scripts in **Project Root**

### 1.1. Dependencies

1. Install Python dependencies
    ```shell
    $ pip install -r requirements.txt
    ```
2. Install `HKToss-Package` package
    ```shell
    $ pip install -e package/
    ```

---

## 2. Run with Docker

-   run this in **Project Root**

```shell
$ docker compose -f docker-compose.airflow.yml up
```

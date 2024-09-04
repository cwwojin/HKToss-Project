#!/bin/bash

mongoimport --uri mongodb+srv://hktoss:woojin-team@hktoss-mongodb.yxtlmue.mongodb.net/mongodatabase \
    --collection dataset_main \
    --headerline \
    --type csv \
    --file /home/hoyoon/HKToss-Project/airflow/.tmp/dataset/dataset_train_main.csv
#!/usr/bin/env bash
set -Eeuo pipefail
if [[ -f ./.env ]]; then
  set -a        # всё, что source-им — экспортируется в окружение
  source ./.env
  set +a
else
  echo "Файл .env не найден рядом со скриптом"; exit 1
fi

export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export AWS_BUCKET_NAME=$AWS_BUCKET_NAME
export DB_DESTINATION_USER=$DB_DESTINATION_USER
export DB_DESTINATION_PASSWORD=$DB_DESTINATION_PASSWORD
export DB_DESTINATION_HOST=$DB_DESTINATION_HOST
export DB_DESTINATION_PORT=$DB_DESTINATION_PORT
export DB_DESTINATION_NAME=$DB_DESTINATION_NAME


mlflow server \
  --backend-store-uri postgresql://mle_20250802_8a8f1a1def_freetrack:85e7460544f8458ba1b5b8c027853732@rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net:6432/playground_mle_20250802_8a8f1a1def \
  --registry-store-uri postgresql://mle_20250802_8a8f1a1def_freetrack:85e7460544f8458ba1b5b8c027853732@rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net:6432/playground_mle_20250802_8a8f1a1def \
  --default-artifact-root s3://s3-student-mle-20250802-8a8f1a1def-freetrack \
  --no-serve-artifacts
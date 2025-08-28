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

mlflow server \
  --backend-store-uri postgresql://$DB_DESTINATION_USER:$DB_DESTINATION_PASSWORD@$DB_DESTINATION_HOST:$DB_DESTINATION_PORT/$DB_DESTINATION_NAME\
    --default-artifact-root s3://$AWS_BUCKET_NAME\
    --no-serve-artifacts
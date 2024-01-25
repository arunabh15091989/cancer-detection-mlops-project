# Dockerfile
FROM tensorflow/serving

ENV MODEL_NAME=cancer_model
ENV MODEL_PATH=/models/$MODEL_NAME

COPY tensorflow_serving_entrypoint.sh /usr/bin/tensorflow_serving_entrypoint.sh

EXPOSE 8501

CMD ["tensorflow_serving_entrypoint.sh"]
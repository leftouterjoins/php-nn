FROM php:8.1.10-cli
VOLUME /app
WORKDIR /app
ENTRYPOINT [ "php", "main.php" ]

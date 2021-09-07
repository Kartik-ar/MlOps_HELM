FROM centos:7
RUN yum install httpd -y
COPY templates/ /var/www/html/
ENTRYPOINT ["httpd","-D","FOREGROUND"]

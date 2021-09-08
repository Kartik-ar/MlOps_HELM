FROM centos:latest
RUN dnf install  python3 -y
RUN pip3 install Flask
RUN pip3 install flask-cors
RUN pip3 install scikit-learn
RUN pip3 install pandas
ADD auto-mpg.csv auto-mpg.csv
ADD car_mileage_prediction.py car_mileage_prediction.py
EXPOSE 5960
ENTRYPOINT ["python3", "car_mileage_prediction.py"]


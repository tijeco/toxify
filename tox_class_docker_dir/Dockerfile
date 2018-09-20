FROM ubuntu:12.04
MAINTAINER Jeff Cole <coleti16@students.ecu.edu>

RUN apt-get install gfortran
RUN apt-get install software-properties-common
RUN add-apt-repository ppa:webupd8team/java
RUN apt-get install oracle-java8-installer
RUN wget https://cran.cnr.berkeley.edu/src/base/R-3/R-3.2.3.tar.gz
RUN cd R-3.2.3.tar.gz
RUN tar xvzf R-3.2.3.tar.gz
RUN cd R-3.2.3
RUN ./configure --with-readline=no --with-x=no

RUN apt-get install -y build-essential python python-dev python-pip
RUN apt-get install pkg-config libnlopt-dev
RUN apt-get install libcurl4-gnutls-dev librtmp-dev
RUN pip install pycurl
RUN apt-get install git-lfs
RUN git-lfs clone https://github.com/rgacesa/ToxClassifier.git

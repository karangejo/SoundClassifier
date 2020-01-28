#!/usr/bin/env bash

dependencies="python3 python3-pip python3-virtualenv aubio-tools virtualenv"

check_and_install(){
	KG_OK=$(dpkg-query -W --showformat='${Status}\n' $1|grep "install ok installed")
	echo Checking for package $1: $PKG_OK
	if [ "" == "$PKG_OK" ]; then
  		echo "package not found. Setting up ${1}."
  		sudo apt-get install $1
	fi
}

for arg in $dependencies;
do
check_and_install $arg
done

pip3 install virtualenv

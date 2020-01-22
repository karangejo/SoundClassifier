#!/usr/bin/env bash

dependencies="python3 python3-pip python3-virtualenv aubio-tools"

check_and_install(){
	KG_OK=$(dpkg-query -W --showformat='${Status}\n' aubio-tools|grep "install ok installed")
	echo Checking for aubio-tools: $PKG_OK
	if [ "" == "$PKG_OK" ]; then
  		echo "aubio-tools package not found. Setting up aubio-tools."
  		sudo apt-get install aubio-tools
	fi
}

for arg in $dependencies;
do
check_and_install $arg
done

pip3 install virtualenv

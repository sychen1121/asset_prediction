install:
	virtualenv --python=python2.7 __
	pip install --user -r requirements.txt
	mkdir -r output/


simulate:
	. __/bin/activate
	python -m model.simulation




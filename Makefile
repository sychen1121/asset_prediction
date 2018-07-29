install:
	virtualenv --python=python2.7 __
	pip install --user -r requirements.txt

run:
	. __/bin/activate
	python -m model.simulation

install:
	virtualenv --python=python2.7 __
	pip install -r requirements.txt

run:
	source __/bin/activate
	python -m model.simulation

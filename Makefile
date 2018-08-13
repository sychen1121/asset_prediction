install:
	virtualenv --python=python2.7 __
	pip install --user -r requirements.txt
	mkdir -r output/exp
	mkdir -r output/model
	mkdir -r output/report
	mkdir -r output/result

update:
	. __/bin/activate
	pip install --user -r requirements.txt

predict:
	. __/bin/activate
	python -m app.entry

simulate:
	. __/bin/activate
	python -m app.simulation

reset:
	rm -rf output/model/*
	rm -rf output/prediction/*
	rm -rf output/report/*




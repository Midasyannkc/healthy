.PHONY: test run
test:
	python3 -m pytest -q tests
run:
	python3 -m healthy.run
